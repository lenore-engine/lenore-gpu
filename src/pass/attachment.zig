const std = @import("std");
const vk = @import("vulkan");
const Context = @import("../device/context.zig").Context;
const image = @import("../object/image.zig");
const memory = @import("../memory/allocator.zig");

pub const FormatError = error{
    NoSupportedDepthFormat,
    NoSupportedColourFormat,
};

pub const CreateError = FormatError || image.InitError;

// Written by the pass and sampled by the post pass, which is the pair that
// decides which formats are candidates at all.
pub const hdr_usage: vk.ImageUsageFlags = .{ .color_attachment_bit = true, .sampled_bit = true };

// Written and never read, so no sampled bit. That is also what allows the
// attachment to be discarded at the end of the pass.
pub const depth_usage: vk.ImageUsageFlags = .{ .depth_stencil_attachment_bit = true };

// The sun shadow map, which is the depth image that does not follow the rule
// above: the bake writes it and the main pass samples it, so it is stored rather
// than discarded and it carries the sampled bit.
pub const shadow_usage: vk.ImageUsageFlags = .{
    .depth_stencil_attachment_bit = true,
    .sampled_bit = true,
};

// In preference order.
//
// A stencil-carrying format is not among them. Nothing here uses stencil, and
// carrying it costs either four bytes more per pixel or a precision drop: the
// two combined formats are 8 and 4 bytes against 4 for plain 32-bit depth.
//
// The float format leads, and against 24-bit unorm that ordering buys nothing
// while the depth range runs from the near plane to the far one. The perspective
// divide already concentrates precision near the eye, and the float's exponent
// spends its own precision in the same place, so the two resolve a distance
// equally well over the whole range.
//
// Reversing the depth range is what separates them: it pairs the float's dense
// end, near zero, with the coarse end of that mapping, and buys orders of
// magnitude. A unorm gains nothing from reversal because its spacing is uniform.
// So the float leads because it is the format that leaves reversal open, and not
// for any advantage it holds unreversed.
pub const depth_candidates = [_]vk.Format{
    .d32_sfloat,
    .x8_d24_unorm_pack32,
    .d16_unorm,
};

// In preference order.
//
// The first is 32 bits per pixel against 64 for the second, and on an
// integrated part the CPU and the GPU share one memory bus, so a full-screen
// target read and written every frame is bandwidth taken from everything else.
// It has no alpha channel, which costs nothing here: the blend reads source
// alpha and the destination's is never sampled.
//
// What the narrower width is worth was measured by putting the wide format
// first and running the same scene from the same framing. The frame slowed by
// a third, entirely in the wait for the device, and no banding appeared at
// either format on a smooth grey gradient, which is where mantissas of 6, 6 and
// 5 bits would show first. Half the bytes on a target the main pass writes and
// both bloom and the post pass read is worth that much on a shared bus.
//
// The narrow format carries no sign. A pass that has to write a negative value
// into this target is what would need the wide one.
pub const hdr_candidates = [_]vk.Format{
    .b10g11r11_ufloat_pack32,
    .r16g16b16a16_sfloat,
};

// The first candidate whose features carry the usage. Takes the features
// alongside the candidates rather than a device, so the preference order can be
// exercised without one.
pub fn firstSupported(
    candidates: []const vk.Format,
    features: []const vk.FormatFeatureFlags,
    usage: vk.ImageUsageFlags,
) ?vk.Format {
    std.debug.assert(candidates.len == features.len);
    for (candidates, features) |format, supported| {
        if (image.formatSupports(usage, supported)) return format;
    }
    return null;
}

// What the device can do with a format in optimal tiling. Every image this
// module creates is optimally tiled, and a linear-tiled one would have to ask a
// different field.
pub fn formatFeatures(context: *const Context, format: vk.Format) vk.FormatFeatureFlags {
    return context.instance.getPhysicalDeviceFormatProperties(
        context.physical_device,
        format,
    ).optimal_tiling_features;
}

fn deviceFeatures(
    context: *const Context,
    candidates: []const vk.Format,
    out: []vk.FormatFeatureFlags,
) void {
    std.debug.assert(candidates.len == out.len);
    for (candidates, out) |format, *features| features.* = formatFeatures(context, format);
}

pub fn depthFormat(context: *const Context) FormatError!vk.Format {
    var features: [depth_candidates.len]vk.FormatFeatureFlags = undefined;
    deviceFeatures(context, &depth_candidates, &features);
    return firstSupported(&depth_candidates, &features, depth_usage) orelse
        error.NoSupportedDepthFormat;
}

// The same preference order as the camera's depth buffer, against the wider
// usage. A device offering a format for one and not the other is why this asks
// again rather than reusing `depthFormat`'s answer.
pub fn shadowFormat(context: *const Context) FormatError!vk.Format {
    var features: [depth_candidates.len]vk.FormatFeatureFlags = undefined;
    deviceFeatures(context, &depth_candidates, &features);
    return firstSupported(&depth_candidates, &features, shadow_usage) orelse
        error.NoSupportedDepthFormat;
}

pub fn hdrFormat(context: *const Context) FormatError!vk.Format {
    var features: [hdr_candidates.len]vk.FormatFeatureFlags = undefined;
    deviceFeatures(context, &hdr_candidates, &features);
    return firstSupported(&hdr_candidates, &features, hdr_usage) orelse
        error.NoSupportedColourFormat;
}

// Both attachments are one per swapchain rather than one per frame in flight,
// and the barriers at the start of the main pass are what order one frame's use
// after the previous frame's.
pub fn createDepth(
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    extent: vk.Extent2D,
) CreateError!image.Image {
    return image.Image.init(context, memory_allocator, .{
        .width = extent.width,
        .height = extent.height,
        .format = try depthFormat(context),
        .usage = depth_usage,
        .kind = .depth,
    });
}

pub fn createHdr(
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    extent: vk.Extent2D,
) CreateError!image.Image {
    return image.Image.init(context, memory_allocator, .{
        .width = extent.width,
        .height = extent.height,
        .format = try hdrFormat(context),
        .usage = hdr_usage,
        .kind = .colour,
    });
}
