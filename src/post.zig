const std = @import("std");
const vk = @import("vulkan");
const Context = @import("context.zig").Context;
const descriptors = @import("descriptors.zig");
const pass = @import("pass.zig");

const Allocator = std.mem.Allocator;

// The pass that puts the frame on the screen: one screen-covering triangle that
// samples the HDR target and writes the presentable image.
//
// It owns no depth and clears nothing. Every pixel of the target is written by
// the triangle, so loading the previous contents would be bandwidth spent on
// values that are all about to be replaced.

// The swapchain image this frame acquired.
pub const Target = struct {
    image: vk.Image,
    view: vk.ImageView,
    extent: vk.Extent2D,
};

// Mirrors set 0 of fullscreen.slang.
pub const bindings = [_]descriptors.Binding{
    .{ .slot = 0, .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
};

pub const Sets = descriptors.Sets(&bindings);

const colour_range = vk.ImageSubresourceRange{
    .aspect_mask = .{ .color_bit = true },
    .base_mip_level = 0,
    .level_count = 1,
    .base_array_layer = 0,
    .layer_count = 1,
};

// Takes the acquired image from whatever the presentation engine left it in to
// something that can be rendered to.
//
// The source scope is empty. The submission recording this waits on the image's
// acquire semaphore, and that wait is what orders this write after the previous
// presentation; naming a stage here as well would claim a second dependency
// that does not exist. See `Frame.Submission`.
pub fn beginBarriers(target: Target) [1]vk.ImageMemoryBarrier2 {
    return .{.{
        .src_stage_mask = .{},
        .src_access_mask = .{},
        .dst_stage_mask = .{ .color_attachment_output_bit = true },
        .dst_access_mask = .{ .color_attachment_write_bit = true },
        .old_layout = .undefined,
        .new_layout = .color_attachment_optimal,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = target.image,
        .subresource_range = colour_range,
    }};
}

// Hands the image to the presentation engine.
//
// The destination scope is empty for the same reason the source scope above is:
// the semaphore this submission signals is what presentation waits on. What
// this barrier is for is the layout, which no semaphore performs.
pub fn endBarriers(target: Target) [1]vk.ImageMemoryBarrier2 {
    return .{.{
        .src_stage_mask = .{ .color_attachment_output_bit = true },
        .src_access_mask = .{ .color_attachment_write_bit = true },
        .dst_stage_mask = .{},
        .dst_access_mask = .{},
        .old_layout = .color_attachment_optimal,
        .new_layout = .present_src_khr,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = target.image,
        .subresource_range = colour_range,
    }};
}

pub fn colourAttachment(target: Target) vk.RenderingAttachmentInfo {
    return .{
        .image_view = target.view,
        .image_layout = .color_attachment_optimal,
        .resolve_mode = .{},
        .resolve_image_layout = .undefined,
        .load_op = .dont_care,
        .store_op = .store,
        .clear_value = .{ .color = .{ .float_32 = .{ 0, 0, 0, 1 } } },
    };
}

// The three vertices the fullscreen triangle is drawn with. It reads no buffer:
// the vertex index alone gives the position.
pub const vertex_count: u32 = 3;

pub fn begin(context: *const Context, command_buffer: vk.CommandBuffer, target: Target) void {
    const barriers = beginBarriers(target);
    context.device.cmdPipelineBarrier2(command_buffer, &.{
        .image_memory_barrier_count = barriers.len,
        .p_image_memory_barriers = &barriers,
    });

    const viewport = vk.Viewport{
        .x = 0,
        .y = 0,
        .width = @floatFromInt(target.extent.width),
        .height = @floatFromInt(target.extent.height),
        .min_depth = 0,
        .max_depth = 1,
    };
    context.device.cmdSetViewport(command_buffer, 0, &.{viewport});
    context.device.cmdSetScissor(command_buffer, 0, &.{.{
        .offset = .{ .x = 0, .y = 0 },
        .extent = target.extent,
    }});

    const colour = colourAttachment(target);
    context.device.cmdBeginRendering(command_buffer, &.{
        .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = target.extent },
        .layer_count = 1,
        .view_mask = 0,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&colour),
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

// What the pass samples, written once per HDR target rather than per frame: the
// target outlives every frame and only a resize replaces it.
pub const Source = struct {
    view: vk.ImageView,
    sampler: vk.Sampler,
};

pub fn write(context: *const Context, sets: *const Sets, source: Source) void {
    const info = vk.DescriptorImageInfo{
        .sampler = source.sampler,
        .image_view = source.view,
        // Not restated: this is the layout the main pass transitions its
        // target into, and a descriptor declaring any other is a validation
        // error at draw time and nothing sooner.
        .image_layout = pass.sampled_layout,
    };
    const writes = [_]vk.WriteDescriptorSet{.{
        .dst_set = sets.set(0),
        .dst_binding = bindings[0].slot,
        .dst_array_element = 0,
        .descriptor_count = 1,
        .descriptor_type = bindings[0].kind,
        .p_image_info = @ptrCast(&info),
        .p_buffer_info = &no_buffers,
        .p_texel_buffer_view = &no_texel_buffers,
    }};
    context.device.updateDescriptorSets(&writes, null);
}

// Vulkan specification, VkWriteDescriptorSet: the members not selected by
// descriptorType are ignored, but the pointers are not optional in the
// structure, so they are given something valid to point at.
const no_buffers = [_]vk.DescriptorBufferInfo{};
const no_texel_buffers = [_]vk.BufferView{};
