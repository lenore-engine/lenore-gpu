const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;

// Only the two offset alignments are read, so the rest of the limits never
// needs a value here.
fn limitsWith(uniform: vk.DeviceSize, storage: vk.DeviceSize) vk.PhysicalDeviceLimits {
    var limits: vk.PhysicalDeviceLimits = undefined;
    limits.min_uniform_buffer_offset_alignment = uniform;
    limits.min_storage_buffer_offset_alignment = storage;
    return limits;
}

test "a payload shorter than the alignment still gets a whole slot" {
    const plan = try gpu.perFrameLayout(3, 10, 4, 64);
    try testing.expectEqual(@as(vk.DeviceSize, 64), plan.stride);
    try testing.expectEqual(@as(vk.DeviceSize, 192), plan.total);
}

test "a payload that is already a multiple of the alignment is not padded" {
    const plan = try gpu.perFrameLayout(2, 32, 4, 64);
    try testing.expectEqual(@as(vk.DeviceSize, 128), plan.stride);
    try testing.expectEqual(@as(vk.DeviceSize, 256), plan.total);

    // One byte more takes a whole further slot: the padding is the price of a
    // dynamic offset the device accepts, not a rounding convenience.
    const spilled = try gpu.perFrameLayout(2, 129, 1, 64);
    try testing.expectEqual(@as(vk.DeviceSize, 192), spilled.stride);
}

test "the last frame's window ends exactly at the end of the buffer" {
    // Vulkan specification, VkWriteDescriptorSet and vkCmdBindDescriptorSets:
    // for a dynamic buffer the sum of the dynamic offset, the descriptor's
    // offset and its range must lie inside the buffer. The descriptor's range
    // is the payload, so this is what makes the highest frame legal.
    const frames = 4;
    const count = 7;
    const element_size = 12;
    const plan = try gpu.perFrameLayout(frames, count, element_size, 256);

    const payload: vk.DeviceSize = count * element_size;
    const last_offset = plan.stride * (frames - 1);
    try testing.expect(last_offset + payload <= plan.total);
    try testing.expectEqual(plan.total, plan.stride * frames);
}

test "an alignment of one leaves the payload exactly as it is" {
    const plan = try gpu.perFrameLayout(2, 3, 5, 1);
    try testing.expectEqual(@as(vk.DeviceSize, 15), plan.stride);
    try testing.expectEqual(@as(vk.DeviceSize, 30), plan.total);
}

test "an empty ring names which dimension was empty" {
    try testing.expectError(error.ZeroFrames, gpu.perFrameLayout(0, 4, 16, 64));
    try testing.expectError(error.ZeroCount, gpu.perFrameLayout(2, 0, 16, 64));
}

test "a size that cannot be expressed is refused rather than wrapped" {
    const max = std.math.maxInt(u64);

    // The element count times the element size.
    try testing.expectError(error.SizeOverflow, gpu.perFrameLayout(1, max, 2, 64));
    // The padding added to a payload already at the top of the range.
    try testing.expectError(error.SizeOverflow, gpu.perFrameLayout(1, max, 1, 64));
    // The stride times the frame count.
    try testing.expectError(error.SizeOverflow, gpu.perFrameLayout(4, max / 2, 1, 1));
}

test "the offset alignment is the strictest the usage demands" {
    const limits = limitsWith(64, 16);

    try testing.expectEqual(
        @as(vk.DeviceSize, 64),
        gpu.perFrameOffsetAlignment(.{ .uniform_buffer_bit = true }, limits, 4),
    );
    try testing.expectEqual(
        @as(vk.DeviceSize, 16),
        gpu.perFrameOffsetAlignment(.{ .storage_buffer_bit = true }, limits, 4),
    );
    try testing.expectEqual(
        @as(vk.DeviceSize, 64),
        gpu.perFrameOffsetAlignment(
            .{ .uniform_buffer_bit = true, .storage_buffer_bit = true },
            limits,
            4,
        ),
    );
}

test "the payload's own alignment is never given up to a laxer device" {
    // A device asking for 16 does not make a slot boundary valid for a type
    // that has to sit on 64: the slice handed out would be misaligned. Both
    // usages are checked, because each folds the device limit in separately.
    try testing.expectEqual(
        @as(vk.DeviceSize, 64),
        gpu.perFrameOffsetAlignment(.{ .storage_buffer_bit = true }, limitsWith(256, 16), 64),
    );
    try testing.expectEqual(
        @as(vk.DeviceSize, 64),
        gpu.perFrameOffsetAlignment(.{ .uniform_buffer_bit = true }, limitsWith(16, 256), 64),
    );

    // With no buffer usage that carries a dynamic offset, the type is the only
    // constraint left.
    try testing.expectEqual(
        @as(vk.DeviceSize, 32),
        gpu.perFrameOffsetAlignment(.{ .vertex_buffer_bit = true }, limitsWith(256, 256), 32),
    );
}

test "the ring's own methods are reached by the compiler" {
    // Every method below needs a device to call, and a test reaches only what
    // it calls, so without this the typed view and the descriptor are never
    // analysed and a change to them surfaces only on the host.
    const Ring = gpu.PerFrame(extern struct { view: [16]f32, position: [4]f32 });
    _ = &Ring.init;
    _ = &Ring.deinit;
    _ = &Ring.slice;
    _ = &Ring.handle;
    _ = &Ring.dynamicOffset;
    _ = &Ring.descriptor;
}
