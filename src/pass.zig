const std = @import("std");
const vk = @import("vulkan");
const Context = @import("context.zig").Context;

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
// frame's use of the same image. That is what these carry; the layouts are
// incidental, because `undefined` as the old layout discards contents that the
// clear is about to replace anyway.
//
// The HDR target is a write-after-read: the previous frame's post pass sampled
// it. Depth is a write-after-write against the previous frame's depth writes,
// and nothing samples it, so no shader stage appears in the source scope.
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
            .src_stage_mask = .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true },
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

// The layout `end` leaves the HDR target in, and therefore the layout anything
// sampling it afterwards has to declare. Named once so the pass and its reader
// cannot state it differently.
pub const sampled_layout: vk.ImageLayout = .shader_read_only_optimal;

// Makes the pass's colour writes available to the post pass sampler. Depth is
// not here because it was never stored.
pub fn endBarriers(target: Target) [1]vk.ImageMemoryBarrier2 {
    return .{.{
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
    }};
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

// Depth is written and never read back, so it is not stored. That is what lets
// the image be lazily allocated and stay in tile memory, which is the point on
// the integrated parts this targets.
//
// The clear is 1.0 because that is the far plane. The camera builds its
// projection with zmath's `perspectiveFovRh`, whose third column is
// `far / (near - far)`: a point on the near plane leaves it with depth 0 and one
// on the far plane with depth 1. zmath keeps the other convention in a separate
// `perspectiveFovRhGl`, which maps to [-1, 1] instead.
pub fn depthAttachment(target: Target) vk.RenderingAttachmentInfo {
    return .{
        .image_view = target.depth_view,
        .image_layout = .depth_attachment_optimal,
        .resolve_mode = .{},
        .resolve_image_layout = .undefined,
        .load_op = .clear,
        .store_op = .dont_care,
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
pub fn begin(
    context: *const Context,
    command_buffer: vk.CommandBuffer,
    target: Target,
    options: Options,
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

    const colour = colourAttachment(target, options);
    const depth = depthAttachment(target);
    context.device.cmdBeginRendering(command_buffer, &.{
        .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = target.extent },
        .layer_count = 1,
        .view_mask = 0,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&colour),
        .p_depth_attachment = &depth,
    });
}

pub fn end(context: *const Context, command_buffer: vk.CommandBuffer, target: Target) void {
    context.device.cmdEndRendering(command_buffer);

    const barriers = endBarriers(target);
    context.device.cmdPipelineBarrier2(command_buffer, &.{
        .image_memory_barrier_count = barriers.len,
        .p_image_memory_barriers = &barriers,
    });
}
