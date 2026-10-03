const std = @import("std");
const vk = @import("vulkan");
const Context = @import("../device/context.zig").Context;

// The main pass renders linear radiance into an HDR target that the post pass
// then samples and tonemaps. The swapchain image is not here: the main pass
// never touches it, and a target carrying attachments its own pass ignores is
// two passes' state in one struct.
pub const Target = struct {
    hdr_image: vk.Image,
    hdr_view: vk.ImageView,
    depth_image: vk.Image,
    depth_view: vk.ImageView,
    extent: vk.Extent2D,
};

pub const Options = struct {
    clear_colour: [4]f32 = .{ 0, 0, 0, 1 },
};

const colour_range = vk.ImageSubresourceRange{
    .aspect_mask = .{ .color_bit = true },
    .base_mip_level = 0,
    .level_count = 1,
    .base_array_layer = 0,
    .layer_count = 1,
};

const depth_range = vk.ImageSubresourceRange{
    .aspect_mask = .{ .depth_bit = true },
    .base_mip_level = 0,
    .level_count = 1,
    .base_array_layer = 0,
    .layer_count = 1,
};

// Both attachments are shared by every frame in flight rather than being one
// per frame, so each frame's writes have to be ordered after the previous
// frame's use of the same image. These run before the depth prepass; the layouts
// are incidental, because `undefined` discards contents both attachments are
// about to replace.
//
// The HDR target is a write-after-read: the previous frame's post pass sampled
// it. Depth is both: a write-after-write against the previous frame's depth
// writes, and a write-after-read against what read it after the previous main
// pass: the application's compute and the post pass's fragment shader.
pub fn beginBarriers(target: Target) [2]vk.ImageMemoryBarrier2 {
    return .{
        .{
            .src_stage_mask = .{ .fragment_shader_bit = true },
            .src_access_mask = .{ .shader_read_bit = true },
            .dst_stage_mask = .{ .color_attachment_output_bit = true },
            .dst_access_mask = .{ .color_attachment_write_bit = true },
            .old_layout = .undefined,
            .new_layout = .color_attachment_optimal,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = target.hdr_image,
            .subresource_range = colour_range,
        },
        .{
            .src_stage_mask = .{
                .early_fragment_tests_bit = true,
                .late_fragment_tests_bit = true,
                .compute_shader_bit = true,
                .fragment_shader_bit = true,
            },
            .src_access_mask = .{ .depth_stencil_attachment_write_bit = true },
            .dst_stage_mask = .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true },
            .dst_access_mask = .{
                .depth_stencil_attachment_read_bit = true,
                .depth_stencil_attachment_write_bit = true,
            },
            .old_layout = .undefined,
            .new_layout = .depth_attachment_optimal,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = target.depth_image,
            .subresource_range = depth_range,
        },
    };
}

// The layout `end` leaves the HDR target and depth in, and therefore the layout
// anything sampling them afterwards has to declare. Named once so the pass and
// its readers cannot state it differently.
pub const sampled_layout: vk.ImageLayout = .shader_read_only_optimal;

// Makes the prepass writes available to depth tests in the main rendering. The
// layout does not change, but dynamic rendering provides no dependency between
// two rendering instances: without this barrier the second may read depth
// before the first has finished writing it.
pub fn prepassBarrier(target: Target) [1]vk.ImageMemoryBarrier2 {
    return .{.{
        .src_stage_mask = .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true },
        .src_access_mask = .{ .depth_stencil_attachment_write_bit = true },
        .dst_stage_mask = .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true },
        .dst_access_mask = .{
            .depth_stencil_attachment_read_bit = true,
            // Application draws that were not part of the prepass may still
            // use the solid pipeline and add a nearer value in the main pass.
            .depth_stencil_attachment_write_bit = true,
        },
        .old_layout = .depth_attachment_optimal,
        .new_layout = .depth_attachment_optimal,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = target.depth_image,
        .subresource_range = depth_range,
    }};
}

// Makes the main rendering's colour writes available to the post pass sampler,
// and its depth writes to both readers after the main pass: the application's
// compute and the post pass.
pub fn endBarriers(target: Target) [2]vk.ImageMemoryBarrier2 {
    return .{
        .{
            .src_stage_mask = .{ .color_attachment_output_bit = true },
            .src_access_mask = .{ .color_attachment_write_bit = true },
            .dst_stage_mask = .{ .fragment_shader_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true },
            .old_layout = .color_attachment_optimal,
            .new_layout = sampled_layout,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = target.hdr_image,
            .subresource_range = colour_range,
        },
        .{
            .src_stage_mask = .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true },
            .src_access_mask = .{ .depth_stencil_attachment_write_bit = true },
            .dst_stage_mask = .{ .compute_shader_bit = true, .fragment_shader_bit = true },
            .dst_access_mask = .{ .shader_sampled_read_bit = true },
            .old_layout = .depth_attachment_optimal,
            .new_layout = sampled_layout,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = target.depth_image,
            .subresource_range = depth_range,
        },
    };
}

pub fn colourAttachment(target: Target, options: Options) vk.RenderingAttachmentInfo {
    return .{
        .image_view = target.hdr_view,
        .image_layout = .color_attachment_optimal,
        .resolve_mode = .{},
        .resolve_image_layout = .undefined,
        .load_op = .clear,
        .store_op = .store,
        .clear_value = .{ .color = .{ .float_32 = options.clear_colour } },
    };
}

// The prepass clears to the far plane and stores the nearest opaque surface for
// the main rendering to load. The store is required across the two dynamic
// rendering instances.
//
// The clear is 1.0 because that is the far plane. The camera builds its
// projection with zmath's `perspectiveFovRh`, whose third column is
// `far / (near - far)`: a point on the near plane leaves it with depth 0 and one
// on the far plane with depth 1. zmath keeps the other convention in a separate
// `perspectiveFovRhGl`, which maps to [-1, 1] instead.
pub fn prepassDepthAttachment(target: Target) vk.RenderingAttachmentInfo {
    return .{
        .image_view = target.depth_view,
        .image_layout = .depth_attachment_optimal,
        .resolve_mode = .{},
        .resolve_image_layout = .undefined,
        .load_op = .clear,
        .store_op = .store,
        .clear_value = .{ .depth_stencil = .{ .depth = 1, .stencil = 0 } },
    };
}

// The value the prepass stored is the input to this rendering, and what it
// leaves is stored for the application to sample after it: application draws
// may have added nearer surfaces the prepass did not see.
pub fn mainDepthAttachment(target: Target) vk.RenderingAttachmentInfo {
    return .{
        .image_view = target.depth_view,
        .image_layout = .depth_attachment_optimal,
        .resolve_mode = .{},
        .resolve_image_layout = .undefined,
        .load_op = .load,
        .store_op = .store,
        .clear_value = .{ .depth_stencil = .{ .depth = 1, .stencil = 0 } },
    };
}

// A positive height, so device y of -1 is the first row. The Vulkan Y flip is
// not here: it is in the matrix the camera ring carries, where a shader that
// inverts it gets the whole transform rather than part of one.
pub fn viewport(extent: vk.Extent2D) vk.Viewport {
    return .{
        .x = 0,
        .y = 0,
        .width = @floatFromInt(extent.width),
        .height = @floatFromInt(extent.height),
        .min_depth = 0,
        .max_depth = 1,
    };
}

pub fn scissor(extent: vk.Extent2D) vk.Rect2D {
    return .{ .offset = .{ .x = 0, .y = 0 }, .extent = extent };
}

// Vulkan specification, vkCmdBeginRendering: the command buffer is recording
// outside a render pass instance, and every attachment view names an image in
// the layout its attachment info declares. The barriers above put them there.
pub fn beginDepthPrepass(
    context: *const Context,
    command_buffer: vk.CommandBuffer,
    target: Target,
) void {
    const barriers = beginBarriers(target);
    context.device.cmdPipelineBarrier2(command_buffer, &.{
        .image_memory_barrier_count = barriers.len,
        .p_image_memory_barriers = &barriers,
    });

    // Viewport and scissor are dynamic state, so they belong to the pass rather
    // than to any pipeline bound inside it.
    context.device.cmdSetViewport(command_buffer, 0, &.{viewport(target.extent)});
    context.device.cmdSetScissor(command_buffer, 0, &.{scissor(target.extent)});

    const depth = prepassDepthAttachment(target);
    context.device.cmdBeginRendering(command_buffer, &.{
        .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = target.extent },
        .layer_count = 1,
        .view_mask = 0,
        .color_attachment_count = 0,
        .p_color_attachments = &no_colour_attachments,
        .p_depth_attachment = &depth,
    });
}

pub fn endDepthPrepass(
    context: *const Context,
    command_buffer: vk.CommandBuffer,
    target: Target,
) void {
    context.device.cmdEndRendering(command_buffer);

    const barriers = prepassBarrier(target);
    context.device.cmdPipelineBarrier2(command_buffer, &.{
        .image_memory_barrier_count = barriers.len,
        .p_image_memory_barriers = &barriers,
    });
}

pub fn beginMain(
    context: *const Context,
    command_buffer: vk.CommandBuffer,
    target: Target,
    options: Options,
) void {
    // Stated again rather than inherited across rendering instances. Dynamic
    // state survives today, but each pass owns the state its pipelines require.
    context.device.cmdSetViewport(command_buffer, 0, &.{viewport(target.extent)});
    context.device.cmdSetScissor(command_buffer, 0, &.{scissor(target.extent)});

    const colour = colourAttachment(target, options);
    const depth = mainDepthAttachment(target);
    context.device.cmdBeginRendering(command_buffer, &.{
        .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = target.extent },
        .layer_count = 1,
        .view_mask = 0,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&colour),
        .p_depth_attachment = &depth,
    });
}

pub fn endMain(context: *const Context, command_buffer: vk.CommandBuffer, target: Target) void {
    context.device.cmdEndRendering(command_buffer);

    const barriers = endBarriers(target);
    context.device.cmdPipelineBarrier2(command_buffer, &.{
        .image_memory_barrier_count = barriers.len,
        .p_image_memory_barriers = &barriers,
    });
}

const no_colour_attachments = [_]vk.RenderingAttachmentInfo{};
