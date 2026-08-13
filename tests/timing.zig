const std = @import("std");
const gpu = @import("lenore-gpu");

const testing = std.testing;

test "each pass boundary of each frame gets a slot of its own" {
    // The whole of the layout. A collision here reads another pass's timestamp
    // and returns a plausible number rather than an error, so the property to
    // pin is that the map is injective over every frame and boundary.
    const frames = 3;
    var seen = std.AutoHashMap(u32, void).init(testing.allocator);
    defer seen.deinit();

    for (0..frames) |frame| {
        for (std.enums.values(gpu.GpuPass)) |pass| {
            for ([_]gpu.GpuTimestampEdge{ .begin, .end }) |edge| {
                const index = gpu.gpuTimestampSlot(frame, pass, edge);
                try testing.expect(index < frames * 8);
                try testing.expect(!seen.contains(index));
                try seen.put(index, {});
            }
        }
    }
    try testing.expectEqual(@as(usize, frames * 8), seen.count());
}

test "a frame's slots are one contiguous run" {
    // The reset and the read address the run by its first slot and a length, so
    // a layout that interleaved frames would clear another frame's results.
    for (0..3) |frame| {
        const first = gpu.gpuTimestampSlot(frame, .shadow, .begin);
        try testing.expectEqual(@as(u32, @intCast(frame * 8)), first);
        const last = gpu.gpuTimestampSlot(frame, .post, .end);
        try testing.expectEqual(first + 7, last);
    }
}

test "a duration is the masked difference in nanoseconds" {
    // One tick per nanosecond makes the arithmetic readable; the period is a
    // multiplier and is tested separately below.
    try testing.expectEqual(@as(u64, 100), gpu.gpuDurationNs(1000, 1100, 64, 1.0));
    try testing.expectEqual(@as(u64, 0), gpu.gpuDurationNs(1000, 1000, 64, 1.0));
}

test "the period scales ticks into nanoseconds" {
    // RADV reports one tick per nanosecond, but the specification does not
    // promise it, and a device reporting a different period would otherwise be
    // read as a different frame time.
    try testing.expectEqual(@as(u64, 200), gpu.gpuDurationNs(0, 100, 64, 2.0));
    try testing.expectEqual(@as(u64, 50), gpu.gpuDurationNs(0, 100, 64, 0.5));
}

test "a counter narrower than sixty-four bits wraps rather than going backwards" {
    // The bits above timestampValidBits are undefined, so both readings are
    // masked before they are subtracted. Across the wrap a plain difference
    // would be enormous, and a signed one negative.
    const bits = 36;
    const modulus: u64 = @as(u64, 1) << bits;
    const begin = modulus - 10;
    const end = 5;
    try testing.expectEqual(@as(u64, 15), gpu.gpuDurationNs(begin, end, bits, 1.0));

    // And the undefined high bits are ignored rather than read as elapsed time.
    const noisy_begin = begin | (@as(u64, 0xABCD) << bits);
    const noisy_end = end | (@as(u64, 0x1234) << bits);
    try testing.expectEqual(@as(u64, 15), gpu.gpuDurationNs(noisy_begin, noisy_end, bits, 1.0));
}

test "a frame with no pass recorded totals zero and names every pass" {
    const frame: gpu.GpuTimings = .{};
    try testing.expectEqual(@as(u64, 0), frame.total());
    for (std.enums.values(gpu.GpuPass)) |pass|
        try testing.expectEqual(@as(u64, 0), frame.get(pass));
}

test "a frame's total is the sum of its passes" {
    var frame: gpu.GpuTimings = .{};
    frame.pass_ns[@intFromEnum(gpu.GpuPass.shadow)] = 400_000;
    frame.pass_ns[@intFromEnum(gpu.GpuPass.main)] = 2_000_000;
    frame.pass_ns[@intFromEnum(gpu.GpuPass.bloom)] = 300_000;
    frame.pass_ns[@intFromEnum(gpu.GpuPass.post)] = 100_000;
    try testing.expectEqual(@as(u64, 2_800_000), frame.total());
    try testing.expectEqual(@as(u64, 2_000_000), frame.get(.main));
}

test "support needs both a width and a period" {
    // A queue family reporting zero valid bits is the specification's way of
    // saying a timestamp cannot be written on it, and asking anyway is a wait
    // for a query nothing will fill.
    try testing.expect((gpu.GpuTimerSupport{ .valid_bits = 36, .period_ns = 1 }).available());
    try testing.expect(!(gpu.GpuTimerSupport{ .valid_bits = 0, .period_ns = 1 }).available());
    try testing.expect(!(gpu.GpuTimerSupport{ .valid_bits = 64, .period_ns = 0 }).available());
}
