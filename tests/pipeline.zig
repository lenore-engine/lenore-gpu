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
        &[_]vk.DynamicState{ .viewport, .scissor, .cull_mode },
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
