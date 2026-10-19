const vk = @import("vulkan");

const Context = @import("../device/context.zig").Context;
const descriptors = @import("../binding/descriptors.zig");
const image = @import("../object/image.zig");
const memory = @import("../memory/allocator.zig");
const pass = @import("scene.zig");

// How far each pixel's surface moved on screen since the last frame, written by
// the camera depth prepass beside the depth: the current position's
// framebuffer coordinate less the previous one's, in UV units. A temporal pass
// reads its history at `uv - motion`.
//
// Written there rather than by the shaded pass, whose pipelines include the
// background's and an application's own draws in a `record` hook, and every
// one of them would have to declare the extra attachment. The prepass draws
// exactly the OPAQUE and MASK geometry, which is the geometry that has one
// position per pixel. A blended surface has none and writes none; the far
// plane keeps the clear, zero, and a reader that needs the sky's motion
// reprojects the far plane through the two cameras.

// Two channels for a vector in UV units. Half precision resolves a motion of a
// thousandth of the screen to about a millionth. Vulkan specification, Required
// Format Support, table "Mandatory Format Support: 16-bit Components":
// R16G16_SFLOAT must be rendered to, sampled and filtered linearly.
pub const format: vk.Format = .r16g16_sfloat;

const usage: vk.ImageUsageFlags = .{ .color_attachment_bit = true, .sampled_bit = true };

// The motion's slot in the scene set, which the renderer joins with the others
// there. The shaded pass reads it for the `motion` debug view.
pub const scene_bindings = [_]descriptors.Binding{
    .{ .slot = 9, .name = "motion_vectors", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
};

// One per swapchain like the depth beside it, and ordered after the previous
// frame's reads by the same barriers.
pub fn create(
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    extent: vk.Extent2D,
) image.InitError!image.Image {
    return image.Image.init(context, memory_allocator, .{
        .width = extent.width,
        .height = extent.height,
        .format = format,
        .usage = usage,
        .kind = .colour,
    });
}

// Points the scene set's slot at the motion. At init and at each resize, when
// no frame is in flight. The reader fetches texels by position, so the sampler
// only has to be a valid one.
pub fn writeResult(context: *const Context, scene_set: vk.DescriptorSet, view: vk.ImageView, sampler: vk.Sampler) void {
    const info = vk.DescriptorImageInfo{ .sampler = sampler, .image_view = view, .image_layout = pass.sampled_layout };
    const writes = [_]vk.WriteDescriptorSet{.{
        .dst_set = scene_set,
        .dst_binding = scene_bindings[0].slot,
        .dst_array_element = 0,
        .descriptor_count = 1,
        .descriptor_type = scene_bindings[0].kind,
        .p_image_info = @ptrCast(&info),
        .p_buffer_info = &no_buffers,
        .p_texel_buffer_view = &no_texel_buffers,
    }};
    context.device.updateDescriptorSets(&writes, null);
}

// The attachment the prepass clears to no motion and stores for its readers.
pub fn attachment(view: vk.ImageView) vk.RenderingAttachmentInfo {
    return .{
        .image_view = view,
        .image_layout = .color_attachment_optimal,
        .resolve_mode = .{},
        .resolve_image_layout = .undefined,
        .load_op = .clear,
        .store_op = .store,
        .clear_value = .{ .color = .{ .float_32 = .{ 0, 0, 0, 0 } } },
    };
}

const range = vk.ImageSubresourceRange{
    .aspect_mask = .{ .color_bit = true },
    .base_mip_level = 0,
    .level_count = 1,
    .base_array_layer = 0,
    .layer_count = 1,
};

// Before the prepass: a write-after-read against the previous frame's shaded
// pass, which sampled it. Undefined as the old layout, since the clear
// replaces all of it.
pub fn beginBarrier(handle: vk.Image) vk.ImageMemoryBarrier2 {
    return .{
        .src_stage_mask = .{ .fragment_shader_bit = true },
        .src_access_mask = .{ .shader_sampled_read_bit = true },
        .dst_stage_mask = .{ .color_attachment_output_bit = true },
        .dst_access_mask = .{ .color_attachment_write_bit = true },
        .old_layout = .undefined,
        .new_layout = .color_attachment_optimal,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = handle,
        .subresource_range = range,
    };
}

// After the prepass: its writes made visible to the shaded pass's fragment
// stage, in the layout the scene set's descriptor names.
pub fn endBarrier(handle: vk.Image) vk.ImageMemoryBarrier2 {
    return .{
        .src_stage_mask = .{ .color_attachment_output_bit = true },
        .src_access_mask = .{ .color_attachment_write_bit = true },
        .dst_stage_mask = .{ .fragment_shader_bit = true },
        .dst_access_mask = .{ .shader_sampled_read_bit = true },
        .old_layout = .color_attachment_optimal,
        .new_layout = pass.sampled_layout,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = handle,
        .subresource_range = range,
    };
}

// Vulkan specification, VkWriteDescriptorSet: the members not selected by
// descriptorType are ignored, but the pointers are not optional.
const no_buffers = [_]vk.DescriptorBufferInfo{};
const no_texel_buffers = [_]vk.BufferView{};
