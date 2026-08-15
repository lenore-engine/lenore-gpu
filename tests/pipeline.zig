const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");
const res = @import("lenore-resources");

const testing = std.testing;

const all_streams = [_]res.VertexStreams{
    .{},
    .{ .skinned = true },
    .{ .colour = true },
    .{ .uv1 = true },
    .{ .skinned = true, .colour = true },
    .{ .skinned = true, .uv1 = true },
    .{ .colour = true, .uv1 = true },
    .{ .skinned = true, .colour = true, .uv1 = true },
};

test "a mesh with no optional stream still binds its positions" {
    const input = gpu.pipelineVertexInput(.{});

    try testing.expectEqual(@as(u32, 1), input.binding_count);
    try testing.expectEqual(@as(u32, 4), input.attribute_count);
    try testing.expectEqual(@as(u32, 0), input.bindings[0].binding);
    try testing.expectEqual(vk.VertexInputRate.vertex, input.bindings[0].input_rate);
}

test "each stream adds its own binding and its own attributes" {
    // Four, two, one and one. A stream contributing the wrong number would
    // leave a location the shader declares unfed, which the pipeline accepts
    // and the draw then reads as undefined.
    try testing.expectEqual(@as(u32, 6), gpu.pipelineVertexInput(.{ .skinned = true }).attribute_count);
    try testing.expectEqual(@as(u32, 5), gpu.pipelineVertexInput(.{ .colour = true }).attribute_count);
    try testing.expectEqual(@as(u32, 5), gpu.pipelineVertexInput(.{ .uv1 = true }).attribute_count);

    const every = gpu.pipelineVertexInput(.{ .skinned = true, .colour = true, .uv1 = true });
    try testing.expectEqual(@as(u32, gpu.Pipeline.max_bindings), every.binding_count);
    try testing.expectEqual(@as(u32, gpu.Pipeline.max_attributes), every.attribute_count);
}

test "no combination of streams collides on a binding or a location" {
    // Vulkan specification, VkPipelineVertexInputStateCreateInfo: binding
    // numbers are distinct across the bindings, and locations are distinct
    // across the attributes. Eight combinations is the whole space, so this is
    // exhaustive rather than a sample.
    for (all_streams) |streams| {
        const input = gpu.pipelineVertexInput(streams);

        for (input.boundStreams(), 0..) |binding, index|
            for (input.boundStreams()[index + 1 ..]) |other|
                try testing.expect(binding.binding != other.binding);

        for (input.declaredAttributes(), 0..) |attribute, index|
            for (input.declaredAttributes()[index + 1 ..]) |other|
                try testing.expect(attribute.location != other.location);
    }
}

test "every attribute names a binding the same input actually binds" {
    // An attribute pointing at an absent binding is the failure a stream
    // present in one list and missing from the other produces.
    for (all_streams) |streams| {
        const input = gpu.pipelineVertexInput(streams);

        for (input.declaredAttributes()) |attribute| {
            var bound = false;
            for (input.boundStreams()) |binding| {
                if (binding.binding == attribute.binding) bound = true;
            }
            try testing.expect(bound);
        }
    }
}

test "a stream's binding carries the stride of the type it feeds" {
    const input = gpu.pipelineVertexInput(.{ .skinned = true });

    // Positions and the skin stream are separate buffers, so a shared stride
    // would step one of them through the other's data.
    try testing.expectEqual(@sizeOf(gpu.GpuVertex), input.bindings[0].stride);
    try testing.expectEqual(@sizeOf(gpu.GpuSkinVertex), input.bindings[1].stride);
    try testing.expect(input.bindings[0].stride != input.bindings[1].stride);
}

test "the masked alpha mode draws with the opaque pipeline" {
    try testing.expectEqual(gpu.PipelineMode.solid, gpu.pipelineModeFor(.@"opaque"));
    try testing.expectEqual(gpu.PipelineMode.solid, gpu.pipelineModeFor(.mask));
    try testing.expectEqual(gpu.PipelineMode.blended, gpu.pipelineModeFor(.blend));
}

test "the pipeline entry points are reached by the compiler" {
    _ = &gpu.Pipeline.create;
    _ = &gpu.Pipeline.createCompute;
    _ = &gpu.Pipeline.createLayout;
    _ = &gpu.Pipeline.createModule;
}

test "positive-determinant glTF faces are counter-clockwise" {
    const state = gpu.Pipeline.rasterizationState(.dynamic, null);

    // `vulkanClip` negates clip y before the positive-height viewport maps it
    // to framebuffer rows. Under Vulkan's signed framebuffer-area convention,
    // the glTF counter-clockwise winding therefore remains counter-clockwise.
    try testing.expectEqual(vk.FrontFace.counter_clockwise, state.front_face);
}

test "scene culling is dynamic and fullscreen culling is fixed off" {
    try testing.expectEqualSlices(
        vk.DynamicState,
        &[_]vk.DynamicState{ .viewport, .scissor, .cull_mode, .front_face },
        gpu.Pipeline.dynamicStates(.dynamic),
    );
    try testing.expectEqualSlices(
        vk.DynamicState,
        &[_]vk.DynamicState{ .viewport, .scissor },
        gpu.Pipeline.dynamicStates(.{ .fixed = .{} }),
    );

    const fixed_back = gpu.Pipeline.rasterizationState(.{ .fixed = .{ .back_bit = true } }, null);
    try testing.expect(fixed_back.cull_mode.back_bit);
    const fixed_none = gpu.Pipeline.rasterizationState(.{ .fixed = .{} }, null);
    try testing.expectEqual(@as(u32, 0), fixed_none.cull_mode.toInt());
}

test "only the opaque pipeline adds to depth, and both read it" {
    const solid = gpu.pipelineDepthStencilState(.solid);
    const blended = gpu.pipelineDepthStencilState(.blended);

    try testing.expectEqual(vk.Bool32.true, solid.depth_test_enable);
    try testing.expectEqual(vk.Bool32.true, blended.depth_test_enable);
    try testing.expectEqual(vk.Bool32.true, solid.depth_write_enable);
    try testing.expectEqual(vk.Bool32.false, blended.depth_write_enable);

    // Less, not less-or-equal: the near plane is 0 and a fragment at the same
    // depth as one already drawn does not replace it.
    try testing.expectEqual(vk.CompareOp.less, solid.depth_compare_op);
    try testing.expectEqual(solid.depth_compare_op, blended.depth_compare_op);
}

test "the background passes at the far plane and leaves no depth behind" {
    const background = gpu.pipelineDepthStencilState(.background);

    // The one mode that has to accept equality. It draws at depth one, the
    // pixels it belongs in are the ones the main pass cleared to that same
    // value, and under a strict comparison every one of them fails: the
    // background would be a pass that draws nothing whatever it samples.
    try testing.expectEqual(vk.CompareOp.less_or_equal, background.depth_compare_op);
    try testing.expectEqual(vk.Bool32.true, background.depth_test_enable);

    // The test is still what keeps it behind the geometry, so switching it off
    // would paint over the scene rather than around it.
    try testing.expectEqual(vk.Bool32.false, background.depth_write_enable);

    // Nothing blends. The background is the first thing in the target wherever
    // it passes, so there is nothing underneath to composite against.
    try testing.expectEqual(vk.Bool32.false, gpu.pipelineBlendAttachment(.background).blend_enable);
}

test "blending is the only thing the blend state changes between modes" {
    const solid = gpu.pipelineBlendAttachment(.solid);
    const blended = gpu.pipelineBlendAttachment(.blended);

    try testing.expectEqual(vk.Bool32.false, solid.blend_enable);
    try testing.expectEqual(vk.Bool32.true, blended.blend_enable);
    try testing.expectEqual(vk.BlendFactor.src_alpha, blended.src_color_blend_factor);
    try testing.expectEqual(vk.BlendFactor.one_minus_src_alpha, blended.dst_color_blend_factor);
    try testing.expectEqual(solid.color_write_mask, blended.color_write_mask);
}

test "a described name stops at the padding the specification defines" {
    // VkPipelineExecutableStatisticKHR carries its name as a fixed array padded
    // with zeroes, not as a slice, so a reader that takes the whole array gets
    // the padding with it.
    var field: [8]u8 = @splat(0);
    @memcpy(field[0..4], "VGPR");
    try std.testing.expectEqualStrings("VGPR", gpu.pipelineDescribedName(&field));

    // No padding at all: the name fills the array and there is nothing to trim.
    const full = "abcdefgh".*;
    try std.testing.expectEqualStrings("abcdefgh", gpu.pipelineDescribedName(&full));

    // Empty is a name of no characters rather than the whole array.
    const empty: [8]u8 = @splat(0);
    try std.testing.expectEqualStrings("", gpu.pipelineDescribedName(&empty));
}

test "the overlay composites premultiplied source and keeps the destination's coverage" {
    const overlay = gpu.pipelineBlendAttachment(.overlay);
    const blended = gpu.pipelineBlendAttachment(.blended);

    // The source axis. These vertices carry colour already multiplied by their
    // coverage, so asking the blend unit to multiply again would darken every
    // translucent pixel by its own alpha.
    try testing.expectEqual(vk.Bool32.true, overlay.blend_enable);
    try testing.expectEqual(vk.BlendFactor.one, overlay.src_color_blend_factor);
    try testing.expectEqual(vk.BlendFactor.src_alpha, blended.src_color_blend_factor);

    // The destination axis, and the whole reason this mode needs a device
    // feature: subpixel text covers each colour channel by a different amount,
    // so what the destination is scaled down by is a colour and not an alpha.
    // The second source output carries it, and for everything that is not text
    // it holds the alpha in all three channels, which is what makes this the
    // same arithmetic as `blended` without a second pipeline.
    try testing.expectEqual(vk.BlendFactor.one_minus_src1_color, overlay.dst_color_blend_factor);
    try testing.expectEqual(vk.BlendFactor.one_minus_src_alpha, blended.dst_color_blend_factor);

    // And the alpha axis: the modes that write an opaque target replace its
    // alpha, which drawing on top of one must not do.
    try testing.expectEqual(vk.BlendFactor.one_minus_src1_alpha, overlay.dst_alpha_blend_factor);
    try testing.expectEqual(vk.BlendFactor.zero, blended.dst_alpha_blend_factor);
}

// Both factors have to name the second source or neither does. A mode that
// scaled the colour by `src1` and the alpha by `src` would composite the
// channels against one coverage and the alpha against another, which is a
// picture that is wrong by a fraction of a pixel at every glyph edge and looks
// almost right.
test "the overlay reads the second source on both axes" {
    const overlay = gpu.pipelineBlendAttachment(.overlay);
    const uses_src1 = [_]vk.BlendFactor{ .one_minus_src1_color, .one_minus_src1_alpha };

    var colour_found = false;
    var alpha_found = false;
    for (uses_src1) |factor| {
        if (overlay.dst_color_blend_factor == factor) colour_found = true;
        if (overlay.dst_alpha_blend_factor == factor) alpha_found = true;
    }
    try testing.expect(colour_found and alpha_found);

    // And no other mode does, because none of them writes a second output.
    for ([4]gpu.PipelineMode{ .solid, .blended, .background, .additive }) |mode| {
        const attachment = gpu.pipelineBlendAttachment(mode);
        for (uses_src1) |factor| {
            try testing.expect(attachment.dst_color_blend_factor != factor);
            try testing.expect(attachment.dst_alpha_blend_factor != factor);
            try testing.expect(attachment.src_color_blend_factor != factor);
            try testing.expect(attachment.src_alpha_blend_factor != factor);
        }
    }
}

test "the two modes that declare no depth attachment test nothing" {
    // `create` reaches the depth state only when the rendering declares a depth
    // attachment. Neither of these does, so what they say about depth has to be
    // inert rather than merely unused.
    for ([2]gpu.PipelineMode{ .additive, .overlay }) |mode| {
        const state = gpu.pipelineDepthStencilState(mode);
        try testing.expectEqual(vk.Bool32.false, state.depth_test_enable);
        try testing.expectEqual(vk.Bool32.false, state.depth_write_enable);
        try testing.expectEqual(vk.CompareOp.always, state.depth_compare_op);
    }
}

test "a described vertex input reaches the pipeline unchanged" {
    // The generalisation the UI pass needs: an input this module did not derive
    // from a mesh's streams still arrives at creation as it was written.
    var input: gpu.PipelineVertexInput = .none;
    input.binding_count = 1;
    input.bindings[0] = .{ .binding = 0, .stride = 24, .input_rate = .vertex };
    input.attribute_count = 1;
    input.attributes[0] = .{ .location = 0, .binding = 0, .format = .r32g32_sfloat, .offset = 0 };

    const config = gpu.PipelineConfig{
        .mode = .overlay,
        .vertex_input = input,
        .culling = .{ .fixed = .{} },
        .formats = .{ .colour = .b8g8r8a8_srgb },
        .layout = .null_handle,
        .stages = .{ .vertex = .{ .module = .null_handle, .entry_point = "vertexMain" } },
    };
    try testing.expectEqual(1, config.vertex_input.binding_count);
    try testing.expectEqual(24, config.vertex_input.bindings[0].stride);
    try testing.expectEqual(vk.Format.r32g32_sfloat, config.vertex_input.attributes[0].format);

    // Nothing reads a buffer through `none`, which is what a fullscreen stage
    // declares.
    try testing.expectEqual(0, gpu.PipelineVertexInput.none.binding_count);
}
