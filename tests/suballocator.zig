const std = @import("std");
const Suballocator = @import("memory-suballocator").Suballocator;

const testing = std.testing;

test "released ranges coalesce in both release orders" {
    for ([_]bool{ false, true }) |reverse| {
        var suballocator = try Suballocator.init(testing.allocator, 1024);
        defer suballocator.deinit(testing.allocator);

        const first = (try suballocator.allocate(testing.allocator, 100, 64, 1)).?;
        const second = (try suballocator.allocate(testing.allocator, 200, 128, 2)).?;
        try testing.expectEqual(@as(u64, 0), first);
        try testing.expectEqual(@as(u64, 128), second);

        if (reverse) {
            try suballocator.free(2, second, 200);
            try suballocator.free(1, first, 100);
        } else {
            try suballocator.free(1, first, 100);
            try suballocator.free(2, second, 200);
        }

        const whole = (try suballocator.allocate(testing.allocator, 1024, 1, 3)).?;
        try testing.expectEqual(@as(u64, 0), whole);
        try suballocator.free(3, whole, 1024);
        try testing.expect(!suballocator.hasAllocations());
    }
}
test "unknown, altered, and duplicate releases are rejected" {
    var suballocator = try Suballocator.init(testing.allocator, 256);
    defer suballocator.deinit(testing.allocator);

    const offset = (try suballocator.allocate(testing.allocator, 64, 16, 7)).?;
    try testing.expectError(
        error.InvalidAllocation,
        suballocator.free(7, offset, 32),
    );
    try testing.expectError(
        error.InvalidAllocation,
        suballocator.free(8, offset, 64),
    );
    try suballocator.free(7, offset, 64);
    try testing.expectError(
        error.InvalidAllocation,
        suballocator.free(7, offset, 64),
    );
}
test "allocation honors alignment and reports exhaustion without mutation" {
    var suballocator = try Suballocator.init(testing.allocator, 512);
    defer suballocator.deinit(testing.allocator);

    const first = (try suballocator.allocate(testing.allocator, 1, 256, 1)).?;
    const second = (try suballocator.allocate(testing.allocator, 1, 256, 2)).?;
    try testing.expectEqual(@as(u64, 0), first);
    try testing.expectEqual(@as(u64, 256), second);
    try testing.expect((try suballocator.allocate(
        testing.allocator,
        256,
        256,
        3,
    )) == null);

    try suballocator.free(1, first, 1);
    try suballocator.free(2, second, 1);
    const whole = (try suballocator.allocate(testing.allocator, 512, 1, 4)).?;
    try suballocator.free(4, whole, 512);
}

test "invalid sizes and alignments fail explicitly" {
    try testing.expectError(
        error.InvalidSize,
        Suballocator.init(testing.allocator, 0),
    );

    var suballocator = try Suballocator.init(testing.allocator, 64);
    defer suballocator.deinit(testing.allocator);
    try testing.expectError(
        error.SizeOverflow,
        suballocator.allocate(testing.allocator, 0, 1, 1),
    );
    try testing.expectError(
        error.SizeOverflow,
        suballocator.allocate(testing.allocator, 1, 0, 1),
    );
    try testing.expectError(
        error.SizeOverflow,
        suballocator.allocate(testing.allocator, 1, 3, 1),
    );
}

test "metadata growth failure preserves ownership and allocator state" {
    const block_size: u64 = 64;
    const allocation_size: u64 = 1;
    const first_id: u64 = 1;
    const failed_id: u64 = 2;
    const replacement_id: u64 = 3;

    var failing = testing.FailingAllocator.init(testing.allocator, .{});
    const allocator = failing.allocator();
    var suballocator = try Suballocator.init(allocator, block_size);

    const first = (try suballocator.allocate(
        allocator,
        allocation_size,
        1,
        first_id,
    )).?;
    failing.fail_index = failing.alloc_index;
    failing.resize_fail_index = failing.resize_index;
    try testing.expectError(
        error.OutOfMemory,
        suballocator.allocate(allocator, allocation_size, 1, failed_id),
    );

    try suballocator.free(first_id, first, allocation_size);
    const replacement = (try suballocator.allocate(
        allocator,
        block_size,
        1,
        replacement_id,
    )).?;
    try suballocator.free(replacement_id, replacement, block_size);
    suballocator.deinit(allocator);

    try testing.expect(failing.has_induced_failure);
    try testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
}
