const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;
const temporal = gpu.Temporal;

// Device coordinates span two across the target, so a device offset of two
// over the width is one render pixel.
test "the jitter reaches the shader in render pixels" {
    const render: vk.Extent2D = .{ .width = 1920, .height = 1200 };
    const pixels = temporal.jitterPixels(.{ 2.0 / 1920.0 * 0.25, -2.0 / 1200.0 * 0.5 }, render);
    try testing.expectApproxEqAbs(@as(f32, 0.25), pixels[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -0.5), pixels[1], 1e-6);
}

test "the push block carries both extents and the reset" {
    const push = temporal.pushFor(.{ .jitter = .{ 0, 0 } }, .{ .width = 1920, .height = 1200 }, .{ .width = 2560, .height = 1600 }, true);
    try testing.expectEqual([2]f32{ 1920, 1200 }, push.render_size);
    try testing.expectEqual([2]f32{ 2560, 1600 }, push.output_size);
    try testing.expectEqual(@as(u32, 1), push.reset);
    try testing.expectEqual(@as(usize, 32), @sizeOf(temporal.PushConstants));
}

// Each frame reads the history the frame before wrote, and never the one it
// is writing.
test "the history alternates and a frame reads what the last one wrote" {
    for (0..6) |frame| {
        const read = temporal.readIndex(frame);
        const written = 1 - read;
        try testing.expectEqual(written, temporal.readIndex(frame + 1));
    }
}

// The resolved picture and the history are read by bloom and the post pass in
// fragment shaders and by the meter in compute, so both scopes are named.
test "the resolve's outputs are handed to fragment and compute readers" {
    const done = temporal.colourBarrier(.null_handle, .attachment_to_sampled);
    try testing.expectEqual(gpu.mainPassSampledLayout, done.new_layout);
    try testing.expect(done.dst_stage_mask.fragment_shader_bit and done.dst_stage_mask.compute_shader_bit);
    const first = temporal.colourBarrier(.null_handle, .undefined_to_sampled);
    try testing.expectEqual(vk.ImageLayout.undefined, first.old_layout);
    try testing.expectEqual(gpu.mainPassSampledLayout, first.new_layout);
}
