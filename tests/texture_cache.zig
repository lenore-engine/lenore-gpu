const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;

test "a tightly packed image accepts both eight-bit colour interpretations" {
    const pixels = [_]u8{
        255, 0,   0,   255,
        0,   255, 0,   255,
        0,   0,   255, 255,
        255, 255, 255, 255,
    };
    try gpu.TextureCache.validateRaw(
        .{ .width = 2, .height = 2, .bytes = &pixels },
        .r8g8b8a8_srgb,
    );
    try gpu.TextureCache.validateRaw(
        .{ .width = 2, .height = 2, .bytes = &pixels },
        .r8g8b8a8_unorm,
    );
}

test "packed biased directional coefficients are four bytes per texel" {
    const texels = [_]u8{
        0xff, 0x03, 0x08, 0x20,
        0x00, 0x06, 0x00, 0x20,
    };
    try gpu.TextureCache.validateRaw(
        .{ .width = 2, .height = 1, .bytes = &texels },
        .a2b10g10r10_unorm_pack32,
    );
    try testing.expectError(error.PixelLengthMismatch, gpu.TextureCache.validateRaw(
        .{ .width = 1, .height = 1, .bytes = &texels },
        .a2b10g10r10_unorm_pack32,
    ));
}

test "raw validation rejects dimensions and lengths before staging" {
    const texel = [_]u8{ 0, 0, 0, 255 };

    try testing.expectError(error.InvalidExtent, gpu.TextureCache.validateRaw(
        .{ .width = 0, .height = 1, .bytes = &.{} },
        .r8g8b8a8_srgb,
    ));
    try testing.expectError(error.InvalidExtent, gpu.TextureCache.validateRaw(
        .{ .width = 1, .height = 0, .bytes = &.{} },
        .r8g8b8a8_srgb,
    ));
    try testing.expectError(error.PixelLengthMismatch, gpu.TextureCache.validateRaw(
        .{ .width = 1, .height = 1, .bytes = texel[0..3] },
        .r8g8b8a8_srgb,
    ));
    try testing.expectError(error.PixelLengthMismatch, gpu.TextureCache.validateRaw(
        .{ .width = 1, .height = 1, .bytes = &.{ 0, 0, 0, 255, 0 } },
        .r8g8b8a8_srgb,
    ));
}

test "a half-float image is measured by its own texel and not by four bytes" {
    // Two texels of four half-float channels: sixteen bytes, where the same
    // extent in eight-bit colour would be eight.
    const texels = [_]u8{
        0x00, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3c,
        0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x3c,
    };
    try gpu.TextureCache.validateRaw(
        .{ .width = 2, .height = 1, .bytes = &texels },
        .r16g16b16a16_sfloat,
    );

    // The length that would be right if the texel size were the constant four
    // this path used to carry. It is the case that says the size comes from the
    // format, and the one a lighting cache of half-floats would trip over.
    try testing.expectError(error.PixelLengthMismatch, gpu.TextureCache.validateRaw(
        .{ .width = 2, .height = 1, .bytes = texels[0..8] },
        .r16g16b16a16_sfloat,
    ));
}

test "raw validation rejects arithmetic overflow and compressed formats" {
    try testing.expectError(error.PixelLengthOverflow, gpu.TextureCache.validateRaw(
        .{ .width = std.math.maxInt(u32), .height = std.math.maxInt(u32), .bytes = &.{} },
        .r8g8b8a8_srgb,
    ));
    try testing.expectError(error.UnsupportedPixelFormat, gpu.TextureCache.validateRaw(
        .{ .width = 1, .height = 1, .bytes = &.{ 0, 0, 0, 255 } },
        vk.Format.bc7_srgb_block,
    ));
}

test "the raw acquisition path is reached by the compiler" {
    _ = &gpu.TextureCache.acquireRaw;
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
    for ([_]gpu.TextureFallback{ .white, .metallic_roughness, .normal, .black, .directional }) |kind|
        try testing.expectEqual(gpu.ImageShape.texture_2d, kind.shape());
    try testing.expectEqual(gpu.ImageShape.cube, gpu.TextureFallback.black_cube.shape());

    // Radiance is linear. Black hides the difference, so nothing but this test
    // holds the format to what the data means.
    try testing.expectEqual(vk.Format.r8g8b8a8_unorm, gpu.TextureFallback.black_cube.format());
    try testing.expectEqual(vk.Format.r8g8b8a8_srgb, gpu.TextureFallback.white.format());
    try testing.expectEqual(vk.Format.r8g8b8a8_unorm, gpu.TextureFallback.normal.format());
    try testing.expectEqual(
        vk.Format.a2b10g10r10_unorm_pack32,
        gpu.TextureFallback.directional.format(),
    );
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
