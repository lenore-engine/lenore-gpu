const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;

test "a tightly packed RGBA8 image accepts both colour interpretations" {
    const pixels = [_]u8{
        255, 0,   0,   255,
        0,   255, 0,   255,
        0,   0,   255, 255,
        255, 255, 255, 255,
    };
    try gpu.TextureCache.validateRgba8(
        .{ .width = 2, .height = 2, .bytes = &pixels },
        .r8g8b8a8_srgb,
    );
    try gpu.TextureCache.validateRgba8(
        .{ .width = 2, .height = 2, .bytes = &pixels },
        .r8g8b8a8_unorm,
    );
}

test "RGBA8 validation rejects dimensions and lengths before staging" {
    const texel = [_]u8{ 0, 0, 0, 255 };

    try testing.expectError(error.InvalidExtent, gpu.TextureCache.validateRgba8(
        .{ .width = 0, .height = 1, .bytes = &.{} },
        .r8g8b8a8_srgb,
    ));
    try testing.expectError(error.InvalidExtent, gpu.TextureCache.validateRgba8(
        .{ .width = 1, .height = 0, .bytes = &.{} },
        .r8g8b8a8_srgb,
    ));
    try testing.expectError(error.PixelLengthMismatch, gpu.TextureCache.validateRgba8(
        .{ .width = 1, .height = 1, .bytes = texel[0..3] },
        .r8g8b8a8_srgb,
    ));
    try testing.expectError(error.PixelLengthMismatch, gpu.TextureCache.validateRgba8(
        .{ .width = 1, .height = 1, .bytes = &.{ 0, 0, 0, 255, 0 } },
        .r8g8b8a8_srgb,
    ));
}

test "RGBA8 validation rejects arithmetic overflow and compressed formats" {
    try testing.expectError(error.PixelLengthOverflow, gpu.TextureCache.validateRgba8(
        .{ .width = std.math.maxInt(u32), .height = std.math.maxInt(u32), .bytes = &.{} },
        .r8g8b8a8_srgb,
    ));
    try testing.expectError(error.UnsupportedPixelFormat, gpu.TextureCache.validateRgba8(
        .{ .width = 1, .height = 1, .bytes = &.{ 0, 0, 0, 255 } },
        vk.Format.bc7_srgb_block,
    ));
}

test "the decoded-pixel acquisition path is reached by the compiler" {
    _ = &gpu.TextureCache.acquireRgba8;
}

// The parser and the image module have separate enums for what a file is, and
// nothing but this mapping ties them together. A swap compiles, creates a 2D
// image for a cube file, and fails only at the descriptor write on a device.
test "a container kind maps to the image shape of the same name" {
    try testing.expectEqual(gpu.ImageShape.texture_2d, gpu.ktx2ImageShape(.texture_2d));
    try testing.expectEqual(gpu.ImageShape.cube, gpu.ktx2ImageShape(.cube));
}

// The fallback's shape decides which sampler declaration it can stand in for.
// A cube fallback created as a 2D image compiles, uploads and then fails at the
// descriptor write, naming the binding rather than the fallback.
test "only the environment fallback is a cube, and it is linear" {
    for ([_]gpu.TextureFallback{ .white, .metallic_roughness, .normal, .black }) |kind|
        try testing.expectEqual(gpu.ImageShape.texture_2d, kind.shape());
    try testing.expectEqual(gpu.ImageShape.cube, gpu.TextureFallback.black_cube.shape());

    // Radiance is linear. Black hides the difference, so nothing but this test
    // holds the format to what the data means.
    try testing.expectEqual(vk.Format.r8g8b8a8_unorm, gpu.TextureFallback.black_cube.format());
    try testing.expectEqual(vk.Format.r8g8b8a8_srgb, gpu.TextureFallback.white.format());
    try testing.expectEqual(vk.Format.r8g8b8a8_unorm, gpu.TextureFallback.normal.format());
}

test "one resident image binds under two samplers without a second reference" {
    // Asymmetric on purpose: equal extents would pass a bind that swapped them,
    // and a mip count of one would pass a bind that dropped it.
    const resident: gpu.ResidentTexture = .{
        .view = @enumFromInt(0x1234),
        .width = 640,
        .height = 480,
        .mip_levels = 3,
    };

    const linear = resident.bind(@enumFromInt(0xa1));
    const nearest = resident.bind(@enumFromInt(0xb2));

    // What the two bindings share is the image, and it is the image that a
    // reference is held for. Samplers come from their own cache.
    try testing.expectEqual(resident, linear.resident());
    try testing.expectEqual(resident, nearest.resident());
    try testing.expect(linear.sampler != nearest.sampler);

    try testing.expectEqual(@as(u32, 640), linear.width);
    try testing.expectEqual(@as(u32, 480), linear.height);
    try testing.expectEqual(@as(u32, 3), linear.mip_levels);
    try testing.expectEqual(resident.view, linear.view);
}
