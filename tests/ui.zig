const std = @import("std");
const gpu = @import("lenore-gpu");
const res = @import("lenore-resources");
const vk = @import("vulkan");

const testing = std.testing;

const target: vk.Extent2D = .{ .width = 100, .height = 100 };

fn clip(x: f32, y: f32, width: f32, height: f32) res.Rect {
    return .{ .x = x, .y = y, .width = width, .height = height };
}

test "the overlay declares the vertex the draw list is written in" {
    // The bindings and the offsets are read off `res.Vertex2D` rather than
    // written again, so this holds the formats, which are the part that is
    // stated rather than derived.
    const input = gpu.uiVertexInput;
    try testing.expectEqual(1, input.binding_count);
    try testing.expectEqual(@sizeOf(res.Vertex2D), input.bindings[0].stride);
    try testing.expectEqual(vk.VertexInputRate.vertex, input.bindings[0].input_rate);

    try testing.expectEqual(3, input.attribute_count);
    try testing.expectEqual(vk.Format.r32g32_sfloat, input.attributes[0].format);
    try testing.expectEqual(@offsetOf(res.Vertex2D, "position"), input.attributes[0].offset);
    try testing.expectEqual(vk.Format.r32g32_sfloat, input.attributes[1].format);
    try testing.expectEqual(@offsetOf(res.Vertex2D, "uv"), input.attributes[1].offset);
    // Four halves, which is what the premultiplied colour is.
    try testing.expectEqual(vk.Format.r16g16b16a16_sfloat, input.attributes[2].format);
    try testing.expectEqual(@offsetOf(res.Vertex2D, "colour"), input.attributes[2].offset);

    // Every attribute names the one binding, and the locations are dense from
    // zero, which is what the shader declares against.
    for (input.attributes[0..input.attribute_count], 0..) |attribute, location| {
        try testing.expectEqual(0, attribute.binding);
        try testing.expectEqual(location, attribute.location);
    }
}

// The two are one decision and they live in two repositories, so what keeps
// them together is that this one is derived and this test says which value it
// derived to. A width changed in `lenore-resources` and not here would bind an
// index buffer that describes the wrong stride, and the picture would be
// geometry read from the middle of itself.
test "the bound index width follows the declaration rather than restating it" {
    try testing.expectEqual(vk.IndexType.uint32, gpu.uiIndexType);
    try testing.expectEqual(u32, res.DrawIndex);
}

test "the vertex stage alone reads the push block" {
    try testing.expectEqual(8, @sizeOf(gpu.UiPushConstants));
    try testing.expectEqual(@sizeOf(gpu.UiPushConstants), gpu.uiPushConstantRange.size);
    try testing.expectEqual(0, gpu.uiPushConstantRange.offset);
    try testing.expect(gpu.uiPushConstantRange.stage_flags.vertex_bit);
    try testing.expect(!gpu.uiPushConstantRange.stage_flags.fragment_bit);
}

test "the shader is handed the reciprocal, so a vertex costs no divide" {
    const inverse = gpu.uiInverseExtent(.{ .width = 1280, .height = 720 });
    try testing.expectApproxEqAbs(1.0 / 1280.0, inverse[0], 1e-9);
    try testing.expectApproxEqAbs(1.0 / 720.0, inverse[1], 1e-9);
}

test "a clip inside the target becomes exactly itself" {
    const scissor = gpu.uiScissorFor(clip(10, 20, 30, 40), target).?;
    try testing.expectEqual(10, scissor.offset.x);
    try testing.expectEqual(20, scissor.offset.y);
    try testing.expectEqual(30, scissor.extent.width);
    try testing.expectEqual(40, scissor.extent.height);
}

test "a fractional clip rounds outward, never inward" {
    // The same conservative rule the producer converts a filled rectangle with.
    // Rounding inward here would clip the edge pixel of geometry the clip is
    // meant to admit, which is a seam along every panel under a fractional
    // content scale.
    const scissor = gpu.uiScissorFor(clip(1.5, 1.5, 3, 3), target).?;
    try testing.expectEqual(1, scissor.offset.x);
    try testing.expectEqual(1, scissor.offset.y);
    try testing.expectEqual(4, scissor.extent.width);
    try testing.expectEqual(4, scissor.extent.height);

    const left: f32 = @floatFromInt(scissor.offset.x);
    const right = left + @as(f32, @floatFromInt(scissor.extent.width));
    try testing.expect(left <= 1.5);
    try testing.expect(right >= 4.5);
}

test "a clip hanging off an edge is clamped, not made negative" {
    // VkRect2D carries a signed origin, so a negative one is representable and
    // is exactly what a scissor may not have.
    const over_left = gpu.uiScissorFor(clip(-10, -10, 20, 20), target).?;
    try testing.expectEqual(0, over_left.offset.x);
    try testing.expectEqual(0, over_left.offset.y);
    try testing.expectEqual(10, over_left.extent.width);
    try testing.expectEqual(10, over_left.extent.height);

    const over_right = gpu.uiScissorFor(clip(95, 95, 20, 20), target).?;
    try testing.expectEqual(95, over_right.offset.x);
    try testing.expectEqual(5, over_right.extent.width);
    try testing.expectEqual(5, over_right.extent.height);
}

test "a clip covering no pixel of the target is no draw at all" {
    try testing.expectEqual(null, gpu.uiScissorFor(clip(200, 0, 10, 10), target));
    try testing.expectEqual(null, gpu.uiScissorFor(clip(0, 200, 10, 10), target));
    try testing.expectEqual(null, gpu.uiScissorFor(clip(-50, 0, 10, 10), target));
    try testing.expectEqual(null, gpu.uiScissorFor(clip(0, 0, 0, 10), target));
    try testing.expectEqual(null, gpu.uiScissorFor(clip(0, 0, 10, 0), target));
}

test "a clip that outlived its target is clamped to what the target is now" {
    // The producer validated the clip against the extent it laid out for, and
    // the swapchain can have shrunk between then and here.
    const smaller: vk.Extent2D = .{ .width = 16, .height = 16 };
    const scissor = gpu.uiScissorFor(clip(0, 0, 1000, 1000), smaller).?;
    try testing.expectEqual(0, scissor.offset.x);
    try testing.expectEqual(16, scissor.extent.width);
    try testing.expectEqual(16, scissor.extent.height);
}

test "a non-finite clip yields the target rather than an out-of-range cast" {
    // `@max` and `@min` return the operand that is not NaN, so each edge takes
    // the bound it was compared against. That is what keeps the conversions
    // below in range without a check: in ReleaseFast an `@intFromFloat` of a
    // NaN is undefined behaviour rather than a panic.
    const nan = std.math.nan(f32);
    const scissor = gpu.uiScissorFor(clip(nan, nan, 10, 10), target).?;
    try testing.expectEqual(0, scissor.offset.x);
    try testing.expectEqual(0, scissor.offset.y);
    try testing.expectEqual(100, scissor.extent.width);
    try testing.expectEqual(100, scissor.extent.height);

    const infinite = gpu.uiScissorFor(clip(0, 0, std.math.inf(f32), 10), target).?;
    try testing.expectEqual(100, infinite.extent.width);
    try testing.expectEqual(10, infinite.extent.height);
}

test "the image binding is one sampled image the fragment stage reads" {
    try testing.expectEqual(1, gpu.ui_bindings.len);
    try testing.expectEqual(0, gpu.ui_bindings[0].slot);
    try testing.expectEqual(vk.DescriptorType.combined_image_sampler, gpu.ui_bindings[0].kind);
    try testing.expectEqual(1, gpu.ui_bindings[0].count);
    try testing.expect(gpu.ui_bindings[0].stages.fragment_bit);
    try testing.expect(!gpu.ui_bindings[0].stages.vertex_bit);
}
