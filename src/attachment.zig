const std = @import("std");
const vk = @import("vulkan");
const Context = @import("context.zig").Context;
const image = @import("image.zig");
const memory = @import("memory/allocator.zig");

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

// In preference order.
//
// A stencil-carrying format is not among them. Nothing here uses stencil, and
// carrying it costs either four bytes more per pixel or a precision drop: the
// two combined formats are 8 and 4 bytes against 4 for plain 32-bit depth.
//
// DECIDE: whether 24-bit unorm depth should come first instead. It is the same
// four bytes and spends every bit on depth, where the float format spends eight
// on an exponent, but its precision is spread evenly through the view volume
// rather than concentrated near the eye. What closes this: depth fighting on a
// scene with a far plane at the camera's default 1000, at both formats, forced
// by hand.
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
// DECIDE: whether the narrower format holds up. Its mantissas are 6, 6 and 5
// bits and it cannot carry a negative value, so banding in a dark gradient is
// what would rule it out. What closes this: the same scene at both formats,
// frame time beside a look at a dark smooth surface.
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

fn deviceFeatures(
    context: *const Context,
    candidates: []const vk.Format,
    out: []vk.FormatFeatureFlags,
) void {
    std.debug.assert(candidates.len == out.len);
    for (candidates, out) |format, *features| {
        features.* = context.instance.getPhysicalDeviceFormatProperties(
            context.physical_device,
            format,
        ).optimal_tiling_features;
    }
}

pub fn depthFormat(context: *const Context) FormatError!vk.Format {
    var features: [depth_candidates.len]vk.FormatFeatureFlags = undefined;
    deviceFeatures(context, &depth_candidates, &features);
    return firstSupported(&depth_candidates, &features, depth_usage) orelse
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
