const std = @import("std");
const gpu = @import("lenore-gpu");

const testing = std.testing;
const MaterialData = gpu.MaterialData;
const MaterialInfo = gpu.MaterialInfo;
const TextureSlot = gpu.MaterialTextureSlot;

// Resolving the type is what makes its comptime layout asserts run. Without a
// use like this one they are inert: aliasing a type in root.zig does not resolve
// it, and the shader-facing layout would go unchecked.
test "the packed layout is what the shader expects" {
    try testing.expectEqual(@as(usize, 224), @sizeOf(MaterialData));
    try testing.expectEqual(@as(usize, 16), @alignOf(MaterialData));
    try testing.expectEqual(@as(usize, 0), @offsetOf(MaterialData, "base_colour_factor"));
    try testing.expectEqual(@as(usize, 16), @offsetOf(MaterialData, "emissive_factor"));
    try testing.expectEqual(@as(usize, 32), @offsetOf(MaterialData, "metallic_roughness_cutoff"));
    try testing.expectEqual(@as(usize, 48), @offsetOf(MaterialData, "flags"));
    try testing.expectEqual(@as(usize, 64), @offsetOf(MaterialData, "tex"));
}

test "factors and flags land in their lanes" {
    var info: MaterialInfo = .{
        .name = "packed",
        .textures = .{},
        .factors = .{
            .base_colour = .{ 0.1, 0.2, 0.3, 0.4 },
            .metallic = 0.5,
            .roughness = 0.6,
            .emissive = .{ 0.7, 0.8, 0.9 },
            .normal_scale = 1.5,
            .occlusion_strength = 0.25,
        },
        .rendering = .{ .alpha_mode = .blend, .alpha_cutoff = 0.75, .double_sided = true, .unlit = true },
    };
    const packed_data = MaterialData.fromInfo(&info);

    try testing.expectEqual([4]f32{ 0.1, 0.2, 0.3, 0.4 }, packed_data.base_colour_factor);
    // Occlusion strength rides in the emissive vector's fourth lane.
    try testing.expectEqual([4]f32{ 0.7, 0.8, 0.9, 0.25 }, packed_data.emissive_factor);
    try testing.expectEqual([4]f32{ 0.5, 0.6, 0.75, 1.5 }, packed_data.metallic_roughness_cutoff);

    try testing.expectEqual(@as(u32, 2), packed_data.flags[0]); // blend
    try testing.expectEqual(@as(u32, 1), packed_data.flags[1]); // double sided
    try testing.expectEqual(@as(u32, 0), packed_data.flags[2]); // no textures
    try testing.expectEqual(@as(u32, 1), packed_data.flags[3]); // unlit
}

test "the texture mask reports the four slots that change the shading" {
    var info: MaterialInfo = .{
        .name = "masked",
        .textures = .{},
        .factors = .{},
        .rendering = .{},
    };
    // Base colour deliberately has no bit: a material without one samples the
    // neutral white fallback, which leaves the factor unchanged.
    info.textures.base_colour.path = "base.ktx2";
    try testing.expectEqual(@as(u32, 0), MaterialData.fromInfo(&info).flags[2]);

    info.textures.normal.path = "normal.ktx2";
    info.textures.occlusion.path = "occlusion.ktx2";
    try testing.expectEqual(@as(u32, 0b1001), MaterialData.fromInfo(&info).flags[2]);

    info.textures.metallic_roughness.path = "mr.ktx2";
    info.textures.emissive.path = "emissive.ktx2";
    try testing.expectEqual(@as(u32, 0b1111), MaterialData.fromInfo(&info).flags[2]);
}

test "an absent texture transform packs as the identity" {
    var info: MaterialInfo = .{
        .name = "identity",
        .textures = .{},
        .factors = .{},
        .rendering = .{},
    };
    const packed_data = MaterialData.fromInfo(&info);
    const base = packed_data.tex[@intFromEnum(TextureSlot.base_colour)];

    try testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 1.0 }, base.rs);
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, base.params);
}

// Pins the rotation direction arithmetically, so a change that flips it fails
// here rather than silently in a texture. The visual check is the Khronos
// TextureTransformTest asset, whose rotated quad shows a green marker for the
// correct direction and a red one for the reverse.
test "a quarter turn rotates in the direction UV space needs" {
    var info: MaterialInfo = .{
        .name = "rotated",
        .textures = .{},
        .factors = .{},
        .rendering = .{},
    };
    info.textures.normal.uv = .{
        .set = 1,
        .offset = .{ 0.25, 0.5 },
        .rotation = std.math.pi / 2.0,
        .scale = .{ 2.0, 3.0 },
    };
    const transform = MaterialData.fromInfo(&info).tex[@intFromEnum(TextureSlot.normal)];

    const tolerance = 1e-6;
    try testing.expectApproxEqAbs(@as(f32, 0.0), transform.rs[0], tolerance);
    try testing.expectApproxEqAbs(@as(f32, 3.0), transform.rs[1], tolerance);
    try testing.expectApproxEqAbs(@as(f32, -2.0), transform.rs[2], tolerance);
    try testing.expectApproxEqAbs(@as(f32, 0.0), transform.rs[3], tolerance);

    // Offset is the raw translation, and the UV set travels as a float.
    try testing.expectEqual([4]f32{ 0.25, 0.5, 1.0, 0.0 }, transform.params);
}
