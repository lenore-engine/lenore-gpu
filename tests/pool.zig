const std = @import("std");
const ResourcePool = @import("lenore-gpu").ResourcePool;

const testing = std.testing;

const Pool = ResourcePool(u32);
const Handle = Pool.Handle;

test "a removed entry is unreachable through its handle" {
    var pool: Pool = .empty;
    defer pool.deinit(testing.allocator);

    const handle = try pool.add(testing.allocator, 11);
    try testing.expectEqual(@as(u32, 11), pool.get(handle).?.*);
    try testing.expectEqual(@as(u32, 1), pool.count());

    pool.getMut(handle).?.* = 12;
    try testing.expectEqual(@as(?u32, 12), pool.remove(handle));
    try testing.expectEqual(@as(u32, 0), pool.count());
    try testing.expect(pool.get(handle) == null);
    try testing.expect(pool.getMut(handle) == null);
    try testing.expectEqual(@as(?u32, null), pool.remove(handle));
}

test "the zero handle never resolves" {
    var pool: Pool = .empty;
    defer pool.deinit(testing.allocator);

    try testing.expect(pool.get(.invalid) == null);
    _ = try pool.add(testing.allocator, 1);
    try testing.expect(pool.get(.invalid) == null);
    try testing.expect(pool.getMut(.invalid) == null);
    try testing.expectEqual(@as(?u32, null), pool.remove(.invalid));

    const zeroed: Handle = @enumFromInt(0);
    try testing.expectEqual(Handle.invalid, zeroed);
}

test "a released slot is reused under a new handle" {
    var pool: Pool = .empty;
    defer pool.deinit(testing.allocator);

    const first = try pool.add(testing.allocator, 1);
    _ = pool.remove(first);
    const second = try pool.add(testing.allocator, 2);

    try testing.expect(first != second);
    try testing.expect(pool.get(first) == null);
    try testing.expectEqual(@as(u32, 2), pool.get(second).?.*);
    try testing.expectEqual(@as(u32, 1), pool.count());
}

// The generation is what makes a stale handle fail, and it is finite: reusing
// one slot often enough returns to the generation an outstanding handle still
// holds. Driving the cycle to that point is what the width is chosen to make
// impossible, so this drives a hundred thousand reuses of one slot instead and
// requires the stale handle to be dead through every one of them. That is a
// lower bound on the width and not a measurement of it: it fails on anything up
// to a 16-bit generation and passes on anything above.
test "a stale handle stays dead however often its slot is reused" {
    var pool: Pool = .empty;
    defer pool.deinit(testing.allocator);

    const stale = try pool.add(testing.allocator, 1);
    _ = pool.remove(stale);

    // One slot serves every insert below, which is what advances its
    // generation. An allocator that refuses everything is the proof: taking a
    // new slot needs one and reusing a freed slot needs none.
    var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    const allocator = failing.allocator();

    var cycles: u32 = 0;
    while (cycles < 100_000) : (cycles += 1) {
        const handle = try pool.add(allocator, cycles + 2);
        try testing.expect(handle != stale);
        try testing.expect(pool.get(stale) == null);
        try testing.expectEqual(@as(?u32, cycles + 2), pool.remove(handle));
    }
}

test "the iterator visits live entries and skips released slots" {
    var pool: Pool = .empty;
    defer pool.deinit(testing.allocator);

    const first = try pool.add(testing.allocator, 1);
    const second = try pool.add(testing.allocator, 2);
    const third = try pool.add(testing.allocator, 3);
    _ = pool.remove(second);

    var iterator = pool.iterator();
    try testing.expectEqual(@as(u32, 1), iterator.next().?.*);
    try testing.expectEqual(@as(u32, 3), iterator.next().?.*);
    try testing.expect(iterator.next() == null);

    try testing.expectEqual(@as(u32, 2), pool.count());
    try testing.expectEqual(@as(u32, 1), pool.get(first).?.*);
    try testing.expectEqual(@as(u32, 3), pool.get(third).?.*);
}

test "clearing keeps capacity and drops every entry" {
    var pool: Pool = .empty;
    defer pool.deinit(testing.allocator);

    const first = try pool.add(testing.allocator, 1);
    _ = try pool.add(testing.allocator, 2);
    pool.clearRetainingCapacity();

    try testing.expectEqual(@as(u32, 0), pool.count());
    try testing.expect(pool.get(first) == null);
    var iterator = pool.iterator();
    try testing.expect(iterator.next() == null);

    // Capacity survived, so the refill needs no allocation.
    var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    _ = try pool.add(failing.allocator(), 3);
    try testing.expectEqual(@as(u32, 1), pool.count());
}

test "a failed insert leaves the pool unchanged and remove stays allocation-free" {
    var failing = testing.FailingAllocator.init(testing.allocator, .{});
    const allocator = failing.allocator();

    var pool: Pool = .empty;
    const first = try pool.add(allocator, 1);

    failing.fail_index = failing.alloc_index;
    failing.resize_fail_index = failing.resize_index;

    // Both arrays carry spare capacity from the growth policy, so the insert
    // that fails is the one that first needs to grow, not the next one.
    var value: u32 = 2;
    const failure = while (value < 4096) : (value += 1) {
        _ = pool.add(allocator, value) catch |err| break err;
    } else error.NoFailureInduced;
    try testing.expectEqual(error.OutOfMemory, failure);

    const live = value - 1;
    try testing.expectEqual(live, pool.count());
    try testing.expectEqual(@as(u32, 1), pool.get(first).?.*);

    // Both reservations happen in add, so this release and the insert that
    // reuses its slot run with every further allocation failing.
    try testing.expectEqual(@as(?u32, 1), pool.remove(first));
    const reused = try pool.add(allocator, value);
    try testing.expectEqual(value, pool.get(reused).?.*);
    try testing.expectEqual(live, pool.count());

    pool.deinit(allocator);
    try testing.expect(failing.has_induced_failure);
    try testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
}

test "a slot index is dense, reused, and refused for a stale handle" {
    const allocator = testing.allocator;
    var pool: Pool = .empty;
    defer pool.deinit(allocator);

    // What a registry keeping one descriptor set per slot depends on: the
    // indices the pool hands out fill from zero, so an array sized to the live
    // capacity is always big enough to be indexed by them.
    const first = try pool.add(allocator, 10);
    const second = try pool.add(allocator, 20);
    const third = try pool.add(allocator, 30);
    try testing.expectEqual(@as(?usize, 0), pool.slotIndex(first));
    try testing.expectEqual(@as(?usize, 1), pool.slotIndex(second));
    try testing.expectEqual(@as(?usize, 2), pool.slotIndex(third));

    // A released slot is handed out again, so the parallel array entry is
    // reused with it rather than leaking.
    try testing.expectEqual(@as(?u32, 20), pool.remove(second));
    const replacement = try pool.add(allocator, 40);
    try testing.expectEqual(@as(?usize, 1), pool.slotIndex(replacement));

    // And the handle that named that slot before no longer resolves, which is
    // what stops a stale command from reaching a live descriptor set.
    try testing.expectEqual(@as(?usize, null), pool.slotIndex(second));
    try testing.expectEqual(@as(?usize, null), pool.slotIndex(.invalid));
}
