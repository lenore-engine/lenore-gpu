const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;
const ao = gpu.Ao;

test "the estimate is half the target, rounding up, never empty" {
    try testing.expectEqual(vk.Extent2D{ .width = 960, .height = 540 }, ao.halfExtent(.{ .width = 1920, .height = 1080 }));
    try testing.expectEqual(vk.Extent2D{ .width = 641, .height = 1 }, ao.halfExtent(.{ .width = 1281, .height = 1 }));
}

// The chain halves until a side would drop under eight texels. 1280x800 is
// the half of 2560x1600: level six is 20x12 and level seven would be 10x6.
test "the distance chain stops before a side drops under eight texels" {
    try testing.expectEqual(@as(u32, 7), ao.levelCount(.{ .width = 1280, .height = 800 }));
    try testing.expectEqual(@as(u32, 1), ao.levelCount(.{ .width = 4, .height = 4 }));
    try testing.expectEqual(@as(u32, 1), ao.levelCount(.{ .width = 8192, .height = 8 }));
    try testing.expectEqual(@as(u32, 2), ao.levelCount(.{ .width = 16, .height = 16 }));
    try testing.expectEqual(ao.max_levels, ao.levelCount(.{ .width = 1 << 15, .height = 1 << 15 }));
}

// A level moves alone: a barrier over the whole chain would discard the level
// the next draw reads.
test "a level barrier names one level of the chain" {
    const barrier = ao.levelBarrier(.null_handle, 3, .attachment_to_sampled);
    try testing.expectEqual(@as(u32, 3), barrier.subresource_range.base_mip_level);
    try testing.expectEqual(@as(u32, 1), barrier.subresource_range.level_count);
    try testing.expectEqual(@as(u32, 0), ao.colourBarrier(.null_handle, .attachment_to_sampled).subresource_range.base_mip_level);
}

// The sample spread is measured in the texels being written, so a factor of
// two lost here makes the radius twice the world distance asked for.
test "the push block spaces taps in the half-resolution texels" {
    const frame: ao.Frame = .{
        .half_width = 1.1,
        .half_height = 0.7,
        .depth_scale = -1,
        .depth_offset = -0.05,
        .pixels_per_unit = 800,
        .radius = 0.5,
        .intensity = 1,
        .bias = 0.01,
    };
    const push = ao.pushFor(frame, .{ .width = 400, .height = 200 }, .{ 1, 0 });
    try testing.expectEqual(@as(f32, 400), push.pixels_per_unit);
    try testing.expectEqual([2]f32{ 1.0 / 400.0, 1.0 / 200.0 }, push.texel);
    try testing.expectEqual([2]f32{ 1, 0 }, push.direction);
    try testing.expectEqual(@as(usize, 56), @sizeOf(ao.PushConstants));
}

// The camera depth leaves the prepass as an attachment, is read here, and has
// to be an attachment again for the shaded pass to load. The two barriers have
// to be each other's inverse, or the shaded pass loads depth in a layout it
// did not declare.
test "the camera depth goes to the read layout and back" {
    const handle: vk.Image = .null_handle;
    const read = ao.depthToRead(handle);
    const back = ao.depthToAttachment(handle);
    try testing.expectEqual(vk.ImageLayout.depth_attachment_optimal, read.old_layout);
    try testing.expectEqual(read.new_layout, back.old_layout);
    try testing.expectEqual(vk.ImageLayout.depth_attachment_optimal, back.new_layout);
    try testing.expect(read.src_access_mask.depth_stencil_attachment_write_bit);
    try testing.expect(read.dst_access_mask.shader_sampled_read_bit);
    try testing.expect(back.dst_access_mask.depth_stencil_attachment_write_bit);
    try testing.expect(read.subresource_range.aspect_mask.depth_bit);
}

test "the estimate ends in the layout the main pass samples" {
    const handle: vk.Image = .null_handle;
    const done = ao.colourBarrier(handle, .attachment_to_sampled);
    try testing.expectEqual(gpu.mainPassSampledLayout, done.new_layout);
    const again = ao.colourBarrier(handle, .sampled_to_attachment);
    try testing.expectEqual(done.new_layout, again.old_layout);
    try testing.expectEqual(done.old_layout, again.new_layout);
}

test "the result's scene slot follows the lighting cache's" {
    for (gpu.SceneSetBindings) |binding| {
        if (std.mem.eql(u8, binding.name, "ambient_occlusion")) return;
    }
    return error.SlotMissing;
}
