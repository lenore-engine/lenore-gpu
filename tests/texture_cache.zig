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
