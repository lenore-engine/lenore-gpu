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
// one slot often enough returns to the generation a live handle already holds.
// This drives that cycle on a real pool and pins the count, so the width of
// Generation is a measured property rather than an assumed one.
test "a handle aliases again after a measured number of reuses of its slot" {
    var pool: Pool = .empty;
    defer pool.deinit(testing.allocator);

    const stale = try pool.add(testing.allocator, 1);
    _ = pool.remove(stale);

    var cycles: u32 = 0;
    const collision = while (cycles < 4096) {
        const handle = try pool.add(testing.allocator, cycles + 2);
        cycles += 1;
        if (handle == stale) break handle;
        _ = pool.remove(handle);
    } else null;

    try testing.expect(collision != null);
    try testing.expectEqual(@as(u32, 255), cycles);
    // The occupant is the new one; the stale handle reads it as if it were the
    // original. Only the slot count bounds how long a handle stays safe.
    try testing.expectEqual(@as(u32, 256), pool.get(stale).?.*);
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
