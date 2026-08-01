const std = @import("std");
const gpu = @import("lenore-gpu");

const testing = std.testing;
const MaterialInfo = gpu.MaterialInfo;

// Teardown walks the texture slots by reflection, so the thing worth testing is
// that every slot is reached. A forgotten one shows up as a leak, which is what
// std.testing.allocator fails a test for.
test "teardown frees the name and every texture path" {
    const slot_names = @typeInfo(MaterialInfo.TextureMaps).@"struct".fields;

    var material: MaterialInfo = .{
        .name = try testing.allocator.dupe(u8, "brushed metal"),
        .textures = .{},
        .factors = .{},
        .rendering = .{},
    };
    inline for (slot_names) |field| {
        @field(material.textures, field.name).path =
            try testing.allocator.dupe(u8, "assets/" ++ field.name ++ ".ktx2");
    }

    try testing.expectEqual(@as(usize, 5), slot_names.len);
    material.deinit(testing.allocator);
}

test "a slot without a texture frees nothing" {
    var material: MaterialInfo = .{
        .name = try testing.allocator.dupe(u8, "untextured"),
        .textures = .{},
        .factors = .{},
        .rendering = .{},
    };
    material.textures.base_colour.path = try testing.allocator.dupe(u8, "assets/base.ktx2");
    material.deinit(testing.allocator);
}

test "the defaults are the glTF ones" {
    const material: MaterialInfo = .{
        .name = "",
        .textures = .{},
        .factors = .{},
        .rendering = .{},
    };

    try testing.expectEqual([4]f32{ 1.0, 1.0, 1.0, 1.0 }, material.factors.base_colour);
    try testing.expectEqual(@as(f32, 1.0), material.factors.metallic);
    try testing.expectEqual(@as(f32, 1.0), material.factors.roughness);
    try testing.expectEqual([3]f32{ 0.0, 0.0, 0.0 }, material.factors.emissive);
    try testing.expectEqual(@as(f32, 1.0), material.factors.normal_scale);
    try testing.expectEqual(@as(f32, 1.0), material.factors.occlusion_strength);

    try testing.expectEqual(MaterialInfo.Rendering.AlphaMode.@"opaque", material.rendering.alpha_mode);
    try testing.expectEqual(@as(f32, 0.5), material.rendering.alpha_cutoff);
    try testing.expect(!material.rendering.double_sided);
    try testing.expect(!material.rendering.unlit);

    // An identity transform on UV set zero is what a material without
    // KHR_texture_transform means.
    const uv = material.textures.normal.uv;
    try testing.expectEqual(@as(u32, 0), uv.set);
    try testing.expectEqual([2]f32{ 0.0, 0.0 }, uv.offset);
    try testing.expectEqual(@as(f32, 0.0), uv.rotation);
    try testing.expectEqual([2]f32{ 1.0, 1.0 }, uv.scale);
}
