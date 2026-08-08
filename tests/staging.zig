const std = @import("std");
const placement = @import("staging-placement");

const testing = std.testing;
const BlockPolicy = placement.BlockPolicy;

const policy: BlockPolicy = .{ .block_capacity = 256, .max_blocks = 4 };

fn place(decision: placement.Decision) !placement.Placement {
    return switch (decision) {
        .place => |chosen| chosen,
        else => error.NotPlaced,
    };
}

test "an empty pool asks for its first block" {
    try testing.expectEqual(
        placement.Decision.grow,
        try policy.choose(&.{}, 0, .{ .size = 16, .alignment = 4 }),
    );
}

test "chunks are placed back to back in the current block" {
    const first = try place(try policy.choose(&.{0}, 0, .{ .size = 64, .alignment = 1 }));
    try testing.expectEqual(@as(u32, 0), first.block);
    try testing.expectEqual(@as(u64, 0), first.offset);
    try testing.expectEqual(@as(u64, 64), first.size);

    const second = try place(try policy.choose(&.{64}, 0, .{ .size = 64, .alignment = 1 }));
    try testing.expectEqual(@as(u64, 64), second.offset);
}

test "a chunk starts at its alignment" {
    const wide: BlockPolicy = .{ .block_capacity = 1024, .max_blocks = 1 };

    const chosen = try place(try wide.choose(&.{1}, 0, .{ .size = 16, .alignment = 256 }));
    try testing.expectEqual(@as(u64, 256), chosen.offset);
    try testing.expectEqual(@as(u64, 16), chosen.size);

    // An offset already aligned is not advanced.
    const aligned = try place(try wide.choose(&.{272}, 0, .{ .size = 8, .alignment = 16 }));
    try testing.expectEqual(@as(u64, 272), aligned.offset);
}

// The failure this pool exists to remove. A resource larger than the staging
// space used to be a hard error; now it is a sequence of chunks.
test "a request larger than a block is segmented" {
    const chosen = try place(try policy.choose(
        &.{0},
        0,
        .{ .size = 1000, .alignment = 4, .granularity = 4 },
    ));
    try testing.expectEqual(@as(u64, 0), chosen.offset);
    try testing.expectEqual(@as(u64, 256), chosen.size);
}

// Whole granules, so an image splits between rows and never inside one. The tail
// of a block that cannot hold a row is left unused rather than handed out.
test "a tail shorter than a granule moves to the next block" {
    const request: placement.Request = .{ .size = 64, .alignment = 4, .granularity = 16 };

    const chosen = try place(try policy.choose(&.{ 250, 0 }, 0, request));
    try testing.expectEqual(@as(u32, 1), chosen.block);
    try testing.expectEqual(@as(u64, 0), chosen.offset);
    try testing.expectEqual(@as(u64, 64), chosen.size);

    // With no second block the same request grows the pool instead.
    try testing.expectEqual(
        placement.Decision.grow,
        try policy.choose(&.{250}, 0, request),
    );
}

// The remainder of a request is placed even when it is shorter than a granule,
// because it is the end of the resource rather than a split inside it.
test "a request that is not a multiple of its granularity places its remainder" {
    const request: placement.Request = .{ .size = 250, .alignment = 1, .granularity = 100 };

    const whole = try place(try policy.choose(&.{0}, 0, request));
    try testing.expectEqual(@as(u64, 250), whole.size);

    // Exactly enough is enough, and it is the remainder that proves it: rounding
    // this one down to whole granules would hand back 200 and leave 50 bytes of
    // the block unreachable for the rest of the resource.
    const exact = try place(try policy.choose(&.{6}, 0, request));
    try testing.expectEqual(@as(u64, 250), exact.size);

    // Once it has to be split, only whole granules are handed out.
    const split = try place(try policy.choose(&.{100}, 0, request));
    try testing.expectEqual(@as(u64, 100), split.offset);
    try testing.expectEqual(@as(u64, 100), split.size);
}

// A block capacity that is not a multiple of the alignment can push the aligned
// offset past the end of the block. The subtraction that follows is unsigned and
// its safety check is gone in the shipping build, so what stops it is the
// exhaustion test rather than the mode.
test "an aligned offset past the end of a block exhausts it" {
    const odd: BlockPolicy = .{ .block_capacity = 300, .max_blocks = 1 };

    try testing.expectEqual(
        placement.Decision.flush,
        try odd.choose(&.{300}, 0, .{ .size = 16, .alignment = 256 }),
    );
}

test "growth comes before a stall" {
    const small: BlockPolicy = .{ .block_capacity = 256, .max_blocks = 2 };
    const request: placement.Request = .{ .size = 16, .alignment = 1 };

    try testing.expectEqual(
        placement.Decision.grow,
        try small.choose(&.{256}, 0, request),
    );
    try testing.expectEqual(
        placement.Decision.flush,
        try small.choose(&.{ 256, 256 }, 0, request),
    );
}

// Blocks before the current one are full or hold a tail too short to be worth
// another copy region. Revisiting them would turn one stream into one region per
// block boundary, which is the cost this scan exists to avoid.
test "the scan starts at the current block" {
    const chosen = try place(try policy.choose(
        &.{ 0, 128 },
        1,
        .{ .size = 32, .alignment = 1 },
    ));
    try testing.expectEqual(@as(u32, 1), chosen.block);
    try testing.expectEqual(@as(u64, 128), chosen.offset);
}

// The distinction the caller's retry loop is built on. Growing or flushing is
// answered by asking again; this one never is, and collapsing the two makes the
// loop infinite.
test "a granule wider than a block never fits" {
    try testing.expectError(error.LargerThanBlock, policy.choose(
        &.{0},
        0,
        .{ .size = 512, .alignment = 4, .granularity = 512 },
    ));
    // Empty, full, or at the ceiling makes no difference to it.
    try testing.expectError(error.LargerThanBlock, policy.choose(
        &.{ 256, 256, 256, 256 },
        0,
        .{ .size = 512, .alignment = 4, .granularity = 512 },
    ));
}

// A caller that cannot split a copy says so by asking for its own size as one
// granule, and then it is never handed a chunk it cannot use.
test "an indivisible request is placed whole or not at all" {
    const request: placement.Request = .{ .size = 200, .alignment = 1, .granularity = 200 };

    try testing.expectEqual(
        placement.Decision.grow,
        try policy.choose(&.{100}, 0, request),
    );
    const chosen = try place(try policy.choose(&.{ 100, 0 }, 0, request));
    try testing.expectEqual(@as(u32, 1), chosen.block);
    try testing.expectEqual(@as(u64, 200), chosen.size);
}

test "invalid sizes, alignments and granularities are rejected" {
    try testing.expectError(error.InvalidSize, policy.choose(&.{0}, 0, .{
        .size = 0,
        .alignment = 1,
    }));
    try testing.expectError(error.InvalidGranularity, policy.choose(&.{0}, 0, .{
        .size = 16,
        .alignment = 1,
        .granularity = 0,
    }));
    try testing.expectError(error.InvalidAlignment, policy.choose(&.{0}, 0, .{
        .size = 16,
        .alignment = 3,
    }));
    // Wider than a block: no placement in one can satisfy it, and it is what
    // keeps the aligned offset from overflowing.
    try testing.expectError(error.InvalidAlignment, policy.choose(&.{0}, 0, .{
        .size = 16,
        .alignment = 512,
    }));
}

// Zero passes the int & (int - 1) test in std's isPowerOfTwo, whose guarding
// assert disappears in the shipping build. The explicit zero check is what
// rejects it, and this fails if the two conditions are ever reordered.
test "a zero alignment is rejected with safety off" {
    try testing.expectError(error.InvalidAlignment, policy.choose(&.{0}, 0, .{
        .size = 8,
        .alignment = 0,
    }));
}

test "an alignment equal to the block capacity places at zero" {
    const chosen = try place(try policy.choose(&.{0}, 0, .{ .size = 8, .alignment = 256 }));
    try testing.expectEqual(@as(u64, 0), chosen.offset);

    // Past that offset the block is exhausted, whatever is left of it.
    try testing.expectEqual(
        placement.Decision.grow,
        try policy.choose(&.{8}, 0, .{ .size = 8, .alignment = 256 }),
    );
}
