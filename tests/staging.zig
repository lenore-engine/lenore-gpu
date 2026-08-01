const std = @import("std");
const placement = @import("staging-placement");

const testing = std.testing;
const Bump = placement.Bump;

test "reservations are placed back to back" {
    var bump: Bump = .init(256);

    try testing.expectEqual(@as(u64, 0), try bump.reserve(64, 1));
    try testing.expectEqual(@as(u64, 64), try bump.reserve(64, 1));
    try testing.expectEqual(@as(u64, 128), bump.used);
    try testing.expectEqual(@as(u64, 128), bump.remaining());
}

test "a reservation starts at its alignment" {
    var bump: Bump = .init(1024);

    try testing.expectEqual(@as(u64, 0), try bump.reserve(1, 1));
    // The next request skips to the first aligned offset, and the skipped bytes
    // are consumed with it.
    try testing.expectEqual(@as(u64, 256), try bump.reserve(16, 256));
    try testing.expectEqual(@as(u64, 272), bump.used);

    // An offset already aligned is not advanced.
    try testing.expectEqual(@as(u64, 272), try bump.reserve(8, 16));
}

test "the whole capacity is reservable" {
    var bump: Bump = .init(128);

    try testing.expectEqual(@as(u64, 0), try bump.reserve(128, 1));
    try testing.expectEqual(@as(u64, 0), bump.remaining());
    try testing.expectError(error.OutOfSpace, bump.reserve(1, 1));
}

// The distinction the caller's retry loop is built on. One error means "flush
// and ask again", the other means "never"; collapsing them makes the loop
// infinite for an oversized request.
test "exhaustion is reported as recoverable or not" {
    var bump: Bump = .init(256);

    _ = try bump.reserve(200, 1);
    try testing.expectError(error.OutOfSpace, bump.reserve(100, 1));
    try testing.expectError(error.LargerThanCapacity, bump.reserve(300, 1));

    // A reset answers the first and not the second.
    bump.reset();
    try testing.expectEqual(@as(u64, 0), try bump.reserve(100, 1));
    try testing.expectError(error.LargerThanCapacity, bump.reserve(300, 1));
}

// Alignment padding is what makes the two cases different: a request can fit the
// capacity and still not fit the aligned offset it would start at.
test "a request that fits only after a reset is recoverable" {
    var bump: Bump = .init(256);

    _ = try bump.reserve(1, 1);
    try testing.expectError(error.OutOfSpace, bump.reserve(256, 1));
    bump.reset();
    try testing.expectEqual(@as(u64, 0), try bump.reserve(256, 1));
}

test "invalid sizes and alignments are rejected without consuming space" {
    var bump: Bump = .init(256);

    try testing.expectError(error.InvalidSize, bump.reserve(0, 1));
    try testing.expectError(error.InvalidAlignment, bump.reserve(16, 0));
    try testing.expectError(error.InvalidAlignment, bump.reserve(16, 3));
    // Wider than the region: no placement in it can satisfy it, and it is what
    // keeps the aligned offset from overflowing.
    try testing.expectError(error.InvalidAlignment, bump.reserve(16, 512));

    try testing.expectEqual(@as(u64, 0), bump.used);
    try testing.expectEqual(@as(u64, 0), try bump.reserve(16, 16));
}

// Zero passes the int & (int - 1) test in std's isPowerOfTwo, whose guarding
// assert disappears in the shipping build. The explicit zero check is what
// rejects it, and this fails if the two conditions are ever reordered.
test "a zero alignment is rejected with safety off" {
    var bump: Bump = .init(64);
    try testing.expectError(error.InvalidAlignment, bump.reserve(8, 0));
}

test "an alignment equal to the capacity places at zero" {
    var bump: Bump = .init(64);

    try testing.expectEqual(@as(u64, 0), try bump.reserve(8, 64));
    try testing.expectError(error.OutOfSpace, bump.reserve(8, 64));
}
