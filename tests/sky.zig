const std = @import("std");
const vk = @import("vulkan");
const res = @import("lenore-resources");
const gpu = @import("lenore-gpu");

const testing = std.testing;

// The handles are not dereferenced by anything under test: the configuration is
// data, and what it names is checked rather than what it points at.
const any_layout: vk.PipelineLayout = .null_handle;
const any_module: vk.ShaderModule = .null_handle;
const formats: gpu.PipelineFormats = .{ .colour = .r16g16b16a16_sfloat, .depth = .d32_sfloat };

// A background shader as the module now takes one. The names are deliberately
// not the ones the engine ships: what is under test is that the configuration
// carries through the names it was given, in the stages it was given them for,
// and a spelling shared with the real table would pass whether it did or not.
const background_shader: gpu.SkyShader = .{
    .spirv = &.{},
    .vertex_entry = "aVertexEntry",
    .fragment_entry = "aFragmentEntry",
};

test "the background is configured as the mode that only reads depth" {
    const config = gpu.Sky.config(any_layout, any_module, background_shader, formats);

    // Solid would write depth over the whole frame at the far plane, and
    // blended would composite the environment over the geometry drawn before
    // it. Neither is reported by anything: both draw a picture.
    try testing.expectEqual(gpu.PipelineMode.background, config.mode);

    // Fixed, not dynamic. The scene draws around this one set a cull mode per
    // batch, and a dynamic pipeline here would take whichever one the last of
    // them left behind.
    try testing.expect(config.culling == .fixed);
    try testing.expectEqual(@as(u32, 0), config.culling.fixed.toInt());

    // A depth attachment has to be declared or the depth state is not read at
    // all, and the test is the whole mechanism keeping the background behind
    // the scene.
    try testing.expectEqual(formats.depth, config.formats.depth);
    try testing.expectEqual(formats.colour, config.formats.colour);
}

test "the background reads no vertex buffer and no depth bias" {
    const config = gpu.Sky.config(any_layout, any_module, background_shader, formats);

    // Three vertices out of the vertex index. Declaring a binding that nothing
    // binds is a draw-time error and not a creation-time one.
    try testing.expectEqual(@as(?res.VertexStreams, null), config.streams);
    try testing.expectEqual(@as(u32, 3), gpu.Sky.vertex_count);

    // The bias belongs to the shadow bake. Offsetting a background that is
    // pinned to the far plane could only push it past the clear value it has to
    // compare equal to.
    try testing.expectEqual(@as(?gpu.Pipeline.DepthBias, null), config.depth_bias);
}

test "the background takes both its stages from the sky module" {
    const config = gpu.Sky.config(any_layout, any_module, background_shader, formats);

    try testing.expectEqual(any_module, config.stages.vertex.module);
    try testing.expectEqual(any_layout, config.layout);

    // A fragment stage is not optional here, unlike the opaque half of the
    // bake: this pass writes colour and nothing else in it produces any.
    const fragment = config.stages.fragment orelse return error.MissingFragmentStage;
    try testing.expectEqual(any_module, fragment.module);

    // Each stage takes the name supplied for it and not the other's. A
    // configuration that swapped the two would name entry points the module
    // does carry and fail at creation, which is a device away from this test.
    try testing.expectEqualStrings(
        std.mem.span(background_shader.vertex_entry),
        std.mem.span(config.stages.vertex.entry_point),
    );
    try testing.expectEqualStrings(
        std.mem.span(background_shader.fragment_entry),
        std.mem.span(fragment.entry_point),
    );
}

test "the two backgrounds are the clear colour and the environment" {
    // The absent environment is not a third state. Its cube is black, every
    // term reading it is linear in that sample, and a black background is the
    // picture a scene with no environment has.
    try testing.expectEqual(@as(usize, 2), @typeInfo(gpu.Background).@"enum".fields.len);
    _ = gpu.Background.clear;
    _ = gpu.Background.environment;
}

test "the background is recorded where the two layers meet" {
    // The list is partitioned solid then blended, so the first blended batch is
    // the boundary. Recording after it would composite the transparent surfaces
    // over the environment and then paint the environment on top of them.
    try testing.expectEqual(@as(?usize, 3), gpu.backgroundSlot(3, 7, .environment));
    try testing.expectEqual(@as(?usize, 0), gpu.backgroundSlot(0, 7, .environment));

    // No blended batch: the boundary is the end of the list. Anywhere earlier
    // would draw the background over solid geometry that the depth test would
    // then have to reject it against, which it can, but the draw would also sit
    // before batches whose depth it depends on.
    try testing.expectEqual(@as(?usize, 7), gpu.backgroundSlot(null, 7, .environment));

    // An empty list is that same case with nothing in front of it: the
    // background is the whole picture, which is what an environment with no
    // model loaded should show.
    try testing.expectEqual(@as(?usize, 0), gpu.backgroundSlot(null, 0, .environment));
}

test "a clear background is recorded nowhere at all" {
    // Not a slot past the end, which would still draw. The main pass's clear
    // colour is what stands, and that is the state the conformance assets are
    // judged against.
    try testing.expectEqual(@as(?usize, null), gpu.backgroundSlot(null, 0, .clear));
    try testing.expectEqual(@as(?usize, null), gpu.backgroundSlot(3, 7, .clear));
    try testing.expectEqual(@as(?usize, null), gpu.backgroundSlot(null, 7, .clear));
}
