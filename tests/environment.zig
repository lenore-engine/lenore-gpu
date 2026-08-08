const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;

// The scene set is assembled from two lists written in two files. Nothing but
// this test holds them to one layout: a slot claimed twice is a compile error
// only once the lists are concatenated, and a gap is not an error at all.
test "the environment continues the scene set after the material array" {
    // The joined layout is what the pipeline is built with, so a gap or a
    // collision between the two lists shows up here rather than as a descriptor
    // the shader reads and nobody wrote.
    try testing.expectEqual(@as(usize, 4), gpu.SceneSetBindings.len);
    for (gpu.SceneSetBindings, 0..) |binding, index|
        try testing.expectEqual(@as(u32, @intCast(index)), binding.slot);

    try testing.expectEqual(@as(usize, 1), gpu.MaterialArrayBindings.len);
    try testing.expectEqual(@as(u32, 0), gpu.MaterialArrayBindings[0].slot);

    const expected_slots = [_]u32{ 1, 2, 3 };
    try testing.expectEqual(expected_slots.len, gpu.environment_bindings.len);
    for (gpu.environment_bindings, expected_slots) |binding, slot| {
        try testing.expectEqual(slot, binding.slot);
        try testing.expectEqual(vk.DescriptorType.combined_image_sampler, binding.kind);
        try testing.expectEqual(@as(u32, 1), binding.count);
        // The image-based terms are computed per fragment and nowhere else.
        try testing.expect(binding.stages.fragment_bit);
        try testing.expect(!binding.stages.vertex_bit);
    }
}

// The mip chain of the GGX cubemap is a roughness axis, not a minification
// chain. Sampling it with a nearest mipmap mode returns the roughness step
// nearest the surface's own instead of the interpolation between two, which
// bands across a curved surface rather than failing.
test "the environment sampler interpolates between roughness levels" {
    const config = gpu.environmentSampler;
    try testing.expectEqual(@as(@TypeOf(config.mipmap_mode), .linear), config.mipmap_mode);
    try testing.expectEqual(@as(@TypeOf(config.mag_filter), .linear), config.mag_filter);
    try testing.expectEqual(@as(@TypeOf(config.min_filter), .linear), config.min_filter);

    // The lookup table is a function tabulated over its domain, and a sample at
    // the edge must not wrap to the far side of it. A cube view ignores these.
    try testing.expectEqual(@as(@TypeOf(config.address_mode_u), .clamp_to_edge), config.address_mode_u);
    try testing.expectEqual(@as(@TypeOf(config.address_mode_v), .clamp_to_edge), config.address_mode_v);
    try testing.expectEqual(@as(@TypeOf(config.address_mode_w), .clamp_to_edge), config.address_mode_w);

    // Nothing here is sampled at a level chosen by a screen-space derivative.
    try testing.expect(!config.anisotropic);
}
