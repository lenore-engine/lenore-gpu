const std = @import("std");
const vk = @import("vulkan");
const bloom = @import("bloom.zig");
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

// Runtime look state is supplied by composition for each recording. The pass
// does not retain it, so changing a look cannot mutate work already submitted.
pub const Settings = struct {
    exposure: f32 = 1,
};

pub const SettingsError = error{InvalidExposure};

// The push block the post pipelines are created against, and the record path
// fills. Both entry points read it, and a supplied shader declares this layout
// or is created against a range that does not describe it.
pub const PushConstants = extern struct {
    exposure: f32,
    bloom: f32,
};

pub const push_constant_range: vk.PushConstantRange = .{
    .stage_flags = .{ .fragment_bit = true },
    .offset = 0,
    .size = @sizeOf(PushConstants),
};

// The exposure, checked.
//
// Split out because the bloom chain needs the value before this block can be
// built: the chain applies it on its first step, so the level the composite adds
// is already in the exposed scale and the two passes read one number. The
// alternative was for each to hold its own, which is a picture whose glow and
// whose surfaces disagree about how bright the scene is.
pub fn exposure(settings: Settings) SettingsError!f32 {
    if (!std.math.isFinite(settings.exposure) or settings.exposure < 0)
        return error.InvalidExposure;
    return settings.exposure;
}

// Null is a recording with no chain behind it, and it is not the same as a
// weight of zero: the pipeline built on the entry point that never samples the
// chain is what expresses it. A chain that has not been recorded this frame
// holds whatever its memory held, and an unsigned float format has bit patterns
// that decode to NaN, which multiplication by zero does not remove.
//
// The weight is taken as a `Look` rather than as a float so that the only value
// reaching it is one `bloom.resolve` produced.
pub fn pushConstants(settings: Settings, look: ?bloom.Look) SettingsError!PushConstants {
    return .{
        .exposure = try exposure(settings),
        .bloom = if (look) |resolved| resolved.composite else 0,
    };
}

// The host account of the fragment arithmetic. KhronosGroup/ToneMapping,
// PBR_Neutral defines the operator over non-negative linear Rec. 709 values.
pub fn toneMap(colour: [3]f32, settings: Settings) SettingsError![3]f32 {
    const scale = try exposure(settings);

    var mapped: [3]f32 = undefined;
    inline for (0..3) |channel|
        mapped[channel] = @max(colour[channel] * scale, 0);

    const darkest = @min(mapped[0], @min(mapped[1], mapped[2]));
    const offset = if (darkest < 0.08)
        darkest - 6.25 * darkest * darkest
    else
        0.04;
    inline for (0..3) |channel| mapped[channel] -= offset;

    const start_compression = 0.8 - 0.04;
    const peak = @max(mapped[0], @max(mapped[1], mapped[2]));
    if (peak < start_compression) return mapped;

    const distance_to_white = 1 - start_compression;
    const compressed_peak = 1 - distance_to_white * distance_to_white /
        (peak + distance_to_white - start_compression);
    inline for (0..3) |channel| mapped[channel] *= compressed_peak / peak;

    const desaturation = 1 - 1 / (0.15 * (peak - compressed_peak) + 1);
    inline for (0..3) |channel|
        mapped[channel] = mapped[channel] * (1 - desaturation) + compressed_peak * desaturation;
    return mapped;
}

// The swapchain image this frame acquired.
pub const Target = struct {
    image: vk.Image,
    view: vk.ImageView,
    extent: vk.Extent2D,
};

// The whole descriptor interface a post shader may read: the HDR target the
// main pass wrote, then the bloom chain's finest level.
//
// Slot 1 is written whether or not a recording composites the chain: one set
// serves both pipelines, and the pipeline that does not sample it is the one
// that expresses bloom being off.
pub const bindings = [_]descriptors.Binding{
    .{ .slot = 0, .name = "hdr", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
    .{ .slot = 1, .name = "bloom", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
};

pub const Sets = descriptors.Sets(&bindings);

// What a post shader has to supply for the two present pipelines to be built
// from it.
//
// Two fragment entry points against one vertex stage, and the pair is not a
// switch inside one shader. The compositing entry point names the chain's
// binding and the other never does, so a recording that ran no chain is a
// different pipeline rather than a multiplication by zero over a descriptor
// nothing wrote.
pub const Shader = struct {
    spirv: []const u32,
    vertex_entry: [*:0]const u8,
    fragment_entry: [*:0]const u8,
    bloom_fragment_entry: [*:0]const u8,
};

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
// target outlives every frame and only a resize replaces it. The chain is
// rebuilt by the same resize, which is why both arrive together.
pub const Source = struct {
    view: vk.ImageView,
    sampler: vk.Sampler,
    bloom_view: vk.ImageView,
    bloom_sampler: vk.Sampler,
};

pub fn write(context: *const Context, sets: *const Sets, source: Source) void {
    const infos = [bindings.len]vk.DescriptorImageInfo{
        .{
            .sampler = source.sampler,
            .image_view = source.view,
            // Not restated: this is the layout the main pass transitions its
            // target into, and a descriptor declaring any other is a validation
            // error at draw time and nothing sooner.
            .image_layout = pass.sampled_layout,
        },
        .{
            .sampler = source.bloom_sampler,
            .image_view = source.bloom_view,
            // The same layout, and for the same reason: the chain's last barrier
            // leaves its finest level in it.
            .image_layout = pass.sampled_layout,
        },
    };

    var writes: [bindings.len]vk.WriteDescriptorSet = undefined;
    for (bindings, &infos, &writes) |binding, *info, *write_entry| {
        write_entry.* = .{
            .dst_set = sets.set(0),
            .dst_binding = binding.slot,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = binding.kind,
            .p_image_info = @ptrCast(info),
            .p_buffer_info = &no_buffers,
            .p_texel_buffer_view = &no_texel_buffers,
        };
    }
    context.device.updateDescriptorSets(&writes, null);
}

// Vulkan specification, VkWriteDescriptorSet: the members not selected by
// descriptorType are ignored, but the pointers are not optional in the
// structure, so they are given something valid to point at.
const no_buffers = [_]vk.DescriptorBufferInfo{};
const no_texel_buffers = [_]vk.BufferView{};
