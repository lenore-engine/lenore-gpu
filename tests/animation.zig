const std = @import("std");
const zm = @import("zmath");
const gpu = @import("lenore-gpu");

const testing = std.testing;
const Animation = gpu.Animation;
const Channel = gpu.AnimationChannel;

const tolerance = 1e-5;

fn vectorKeys(times: []const f32, values: []const [3]f32) ![]gpu.Keyframe(zm.Vec) {
    const keys = try testing.allocator.alloc(gpu.Keyframe(zm.Vec), times.len);
    for (keys, times, values) |*key, time, value| {
        key.* = .{ .time = time, .value = zm.f32x4(value[0], value[1], value[2], 0) };
    }
    return keys;
}

fn oneChannel(channel: Channel) ![]Channel {
    const channels = try testing.allocator.alloc(Channel, 1);
    channels[0] = channel;
    return channels;
}

fn expectVector(expected: [3]f32, actual: zm.Vec) !void {
    // Indexing a @Vector needs a comptime-known index, so the lanes are copied
    // into an array and iterated there.
    const lanes: [4]f32 = actual;
    for (expected, lanes[0..3]) |want, got|
        try testing.expectApproxEqAbs(want, got, tolerance);
}

test "a translation track interpolates between its keys" {
    const keys = try vectorKeys(
        &.{ 0.0, 2.0 },
        &.{ .{ 0, 0, 0 }, .{ 10, 0, 0 } },
    );
    var animation = try Animation.init(
        testing.allocator,
        try oneChannel(.{ .track = .{ .translation = keys }, .target_slot = 0, .interpolation = .linear }),
        "slide",
    );
    defer animation.deinit(testing.allocator);

    var translations: [1]zm.Vec = .{zm.f32x4(0, 0, 0, 0)};
    var rotations: [1]zm.Quat = .{zm.qidentity()};
    var scales: [1]zm.Vec = .{zm.f32x4(1, 1, 1, 0)};

    try animation.sample(animation.cursorAt(0.0), &translations, &rotations, &scales);
    try expectVector(.{ 0, 0, 0 }, translations[0]);

    try animation.sample(animation.cursorAt(1.0), &translations, &rotations, &scales);
    try expectVector(.{ 5, 0, 0 }, translations[0]);

    // Looping is the caller's, so the end of the window is the start again only
    // because the cursor is asked for.
    try animation.sample(animation.cursorAt(2.0), &translations, &rotations, &scales);
    try expectVector(.{ 0, 0, 0 }, translations[0]);

    // Sampling past the end without a cursor clamps instead of wrapping, which
    // is what an absolute timeline needs.
    try animation.sample(9.0, &translations, &rotations, &scales);
    try expectVector(.{ 10, 0, 0 }, translations[0]);
}

test "step interpolation holds the left key" {
    const keys = try vectorKeys(
        &.{ 0.0, 2.0 },
        &.{ .{ 0, 0, 0 }, .{ 10, 0, 0 } },
    );
    var animation = try Animation.init(
        testing.allocator,
        try oneChannel(.{ .track = .{ .translation = keys }, .target_slot = 0, .interpolation = .step }),
        "snap",
    );
    defer animation.deinit(testing.allocator);

    var translations: [1]zm.Vec = undefined;
    var rotations: [1]zm.Quat = undefined;
    var scales: [1]zm.Vec = undefined;

    try animation.sample(1.9, &translations, &rotations, &scales);
    try expectVector(.{ 0, 0, 0 }, translations[0]);
}

// A clip whose keys start late must play its keyed window on every iteration,
// not hold the first key for the offset each time round.
test "an offset timeline loops over its keyed window" {
    const keys = try vectorKeys(
        &.{ 10.0, 12.0 },
        &.{ .{ 0, 0, 0 }, .{ 20, 0, 0 } },
    );
    var animation = try Animation.init(
        testing.allocator,
        try oneChannel(.{ .track = .{ .translation = keys }, .target_slot = 0, .interpolation = .linear }),
        "offset",
    );
    defer animation.deinit(testing.allocator);

    try testing.expectEqual(@as(f32, 10.0), animation.start_time);
    try testing.expectEqual(@as(f32, 12.0), animation.duration);
    try testing.expectEqual(@as(f32, 2.0), animation.loopSpan());

    // Elapsed zero maps onto the first key, not onto ten seconds of hold.
    try testing.expectApproxEqAbs(@as(f32, 10.0), animation.cursorAt(0.0), tolerance);
    try testing.expectApproxEqAbs(@as(f32, 11.0), animation.cursorAt(1.0), tolerance);
    // The transform is idempotent for a cursor already inside one span, which
    // is what lets a caller advance by loopSpan without wrapping twice.
    try testing.expectApproxEqAbs(
        animation.cursorAt(1.0),
        animation.cursorAt(animation.cursorAt(1.0) - animation.start_time),
        tolerance,
    );

    var translations: [1]zm.Vec = undefined;
    var rotations: [1]zm.Quat = undefined;
    var scales: [1]zm.Vec = undefined;
    try animation.sample(animation.cursorAt(1.0), &translations, &rotations, &scales);
    try expectVector(.{ 10, 0, 0 }, translations[0]);
}

test "a clip that holds one pose has no loop span to divide by" {
    const keys = try vectorKeys(&.{5.0}, &.{.{ 1, 2, 3 }});
    var animation = try Animation.init(
        testing.allocator,
        try oneChannel(.{ .track = .{ .translation = keys }, .target_slot = 0, .interpolation = .linear }),
        "hold",
    );
    defer animation.deinit(testing.allocator);

    try testing.expectEqual(@as(f32, 0.0), animation.loopSpan());
    try testing.expectEqual(@as(f32, 5.0), animation.cursorAt(1000.0));

    var translations: [1]zm.Vec = undefined;
    var rotations: [1]zm.Quat = undefined;
    var scales: [1]zm.Vec = undefined;
    try animation.sample(animation.cursorAt(1000.0), &translations, &rotations, &scales);
    try expectVector(.{ 1, 2, 3 }, translations[0]);
}

test "a rotation travels the arc between its keys" {
    const keys = try testing.allocator.alloc(gpu.Keyframe(zm.Quat), 2);
    const axis = zm.f32x4(0, 0, 1, 0);
    keys[0] = .{ .time = 0.0, .value = zm.qidentity() };
    keys[1] = .{ .time = 2.0, .value = zm.quatFromAxisAngle(axis, std.math.pi / 2.0) };

    var animation = try Animation.init(
        testing.allocator,
        try oneChannel(.{ .track = .{ .rotation = keys }, .target_slot = 0, .interpolation = .linear }),
        "turn",
    );
    defer animation.deinit(testing.allocator);

    var translations: [1]zm.Vec = undefined;
    var rotations: [1]zm.Quat = undefined;
    var scales: [1]zm.Vec = undefined;

    // Halfway is an eighth of a turn, and the result stays a unit quaternion,
    // which linear blending of quaternions would not preserve.
    try animation.sample(1.0, &translations, &rotations, &scales);
    const expected: [4]f32 = zm.quatFromAxisAngle(axis, std.math.pi / 4.0);
    const sampled: [4]f32 = rotations[0];
    for (expected, sampled) |want, got|
        try testing.expectApproxEqAbs(want, got, tolerance);

    const length = @sqrt(@reduce(.Add, rotations[0] * rotations[0]));
    try testing.expectApproxEqAbs(@as(f32, 1.0), length, tolerance);
}

// Target slots come from an asset, and the shipping build has no bounds check
// behind this one.
test "a target slot outside the caller's arrays is refused" {
    const keys = try vectorKeys(&.{ 0.0, 1.0 }, &.{ .{ 0, 0, 0 }, .{ 1, 0, 0 } });
    var animation = try Animation.init(
        testing.allocator,
        try oneChannel(.{ .track = .{ .translation = keys }, .target_slot = 4, .interpolation = .linear }),
        "distant",
    );
    defer animation.deinit(testing.allocator);

    try testing.expectEqual(@as(u64, 5), animation.slot_count);

    var translations: [2]zm.Vec = undefined;
    var rotations: [2]zm.Quat = undefined;
    var scales: [2]zm.Vec = undefined;
    try testing.expectError(
        error.TargetArrayTooSmall,
        animation.sample(0.0, &translations, &rotations, &scales),
    );

    // A channel naming the largest slot there is must not wrap the count to
    // zero, which would turn the bounds check below into a permission.
    const extreme_keys = try vectorKeys(&.{ 0.0, 1.0 }, &.{ .{ 0, 0, 0 }, .{ 1, 0, 0 } });
    var extreme = try Animation.init(
        testing.allocator,
        try oneChannel(.{
            .track = .{ .translation = extreme_keys },
            .target_slot = std.math.maxInt(u32),
            .interpolation = .linear,
        }),
        "extreme",
    );
    defer extreme.deinit(testing.allocator);
    try testing.expectEqual(@as(u64, std.math.maxInt(u32)) + 1, extreme.slot_count);
    try testing.expectError(
        error.TargetArrayTooSmall,
        extreme.sample(0.0, &translations, &rotations, &scales),
    );
}

test "morph weights blend across their keys" {
    const times = try testing.allocator.alloc(f32, 2);
    times[0] = 0.0;
    times[1] = 2.0;
    const values = try testing.allocator.alloc(f32, 4);
    // Two targets: the first rises, the second falls.
    values[0] = 0.0;
    values[1] = 1.0;
    values[2] = 1.0;
    values[3] = 0.0;

    var animation = try Animation.init(testing.allocator, try oneChannel(.{
        .track = .{ .weights = .{ .times = times, .values = values, .width = 2 } },
        .target_slot = 0,
        .interpolation = .linear,
    }), "blend");
    defer animation.deinit(testing.allocator);

    var weights: [2]f32 = undefined;
    animation.sampleWeights(animation.cursorAt(0.0), &weights);
    try testing.expectApproxEqAbs(@as(f32, 0.0), weights[0], tolerance);
    try testing.expectApproxEqAbs(@as(f32, 1.0), weights[1], tolerance);

    animation.sampleWeights(animation.cursorAt(1.0), &weights);
    try testing.expectApproxEqAbs(@as(f32, 0.5), weights[0], tolerance);
    try testing.expectApproxEqAbs(@as(f32, 0.5), weights[1], tolerance);

    // A transform channel does not answer for weights.
    var translations: [1]zm.Vec = undefined;
    var rotations: [1]zm.Quat = undefined;
    var scales: [1]zm.Vec = undefined;
    try animation.sample(1.0, &translations, &rotations, &scales);
}

test "a clip without a weight track leaves the weights neutral" {
    const keys = try vectorKeys(&.{ 0.0, 1.0 }, &.{ .{ 0, 0, 0 }, .{ 1, 0, 0 } });
    var animation = try Animation.init(
        testing.allocator,
        try oneChannel(.{ .track = .{ .translation = keys }, .target_slot = 0, .interpolation = .linear }),
        "no weights",
    );
    defer animation.deinit(testing.allocator);

    var weights: [3]f32 = .{ 9, 9, 9 };
    animation.sampleWeights(animation.cursorAt(0.5), &weights);
    try testing.expectEqual([3]f32{ 0, 0, 0 }, weights);
}
