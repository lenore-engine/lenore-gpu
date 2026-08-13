const std = @import("std");
const OwningStorage = @import("lenore-gpu").OwningStorage;

const testing = std.testing;

// Stands in for a GPU resource: it has the deinit shape the storage requires and
// records its own destruction so a test can count teardown instead of inferring
// it.
const Tracked = struct {
    tag: u32,
    destroyed: *u32,

    pub fn deinit(self: *Tracked) void {
        self.destroyed.* += 1;
        self.* = undefined;
    }
};

const Storage = OwningStorage(Tracked);

test "removal destroys the value exactly once" {
    var destroyed: u32 = 0;
    var storage: Storage = .empty;
    defer storage.deinit(testing.allocator);

    const handle = try storage.add(testing.allocator, .{ .tag = 1, .destroyed = &destroyed });
    try testing.expectEqual(@as(u32, 1), storage.get(handle).?.tag);

    storage.remove(handle);
    try testing.expectEqual(@as(u32, 1), destroyed);
    try testing.expect(storage.get(handle) == null);
    try testing.expectEqual(@as(u32, 0), storage.count());

    storage.remove(handle);
    try testing.expectEqual(@as(u32, 1), destroyed);
}

test "teardown destroys what is still live and nothing else" {
    var destroyed: u32 = 0;
    var storage: Storage = .empty;

    const first = try storage.add(testing.allocator, .{ .tag = 1, .destroyed = &destroyed });
    _ = try storage.add(testing.allocator, .{ .tag = 2, .destroyed = &destroyed });
    _ = try storage.add(testing.allocator, .{ .tag = 3, .destroyed = &destroyed });
    storage.remove(first);

    storage.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 3), destroyed);
}

// The property that makes add an ownership transfer: the caller has given the
// resource away by the time the slot is reserved, so a failed reservation must
// destroy it rather than hand back a value nobody is holding.
test "a failed insert destroys the value it was given" {
    var destroyed: u32 = 0;
    var failing = testing.FailingAllocator.init(testing.allocator, .{});
    const allocator = failing.allocator();

    var storage: Storage = .empty;
    _ = try storage.add(allocator, .{ .tag = 0, .destroyed = &destroyed });

    failing.fail_index = failing.alloc_index;
    failing.resize_fail_index = failing.resize_index;

    var tag: u32 = 1;
    const failure = while (tag < 4096) : (tag += 1) {
        _ = storage.add(allocator, .{ .tag = tag, .destroyed = &destroyed }) catch |err| break err;
    } else error.NoFailureInduced;
    try testing.expectEqual(error.OutOfMemory, failure);
    try testing.expectEqual(@as(u32, 1), destroyed);

    const live = storage.count();
    storage.deinit(allocator);
    try testing.expectEqual(live + 1, destroyed);
    try testing.expect(failing.has_induced_failure);
    try testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
}

test "lookups and iteration see live values only" {
    var destroyed: u32 = 0;
    var storage: Storage = .empty;
    defer storage.deinit(testing.allocator);

    const first = try storage.add(testing.allocator, .{ .tag = 1, .destroyed = &destroyed });
    const second = try storage.add(testing.allocator, .{ .tag = 2, .destroyed = &destroyed });
    const third = try storage.add(testing.allocator, .{ .tag = 3, .destroyed = &destroyed });

    storage.getMut(second).?.tag = 20;
    try testing.expectEqual(@as(u32, 20), storage.get(second).?.tag);
    storage.remove(second);

    var live = storage.iterator();
    try testing.expectEqual(@as(u32, 1), live.next().?.tag);
    try testing.expectEqual(@as(u32, 3), live.next().?.tag);
    try testing.expect(live.next() == null);

    try testing.expectEqual(@as(u32, 2), storage.count());
    try testing.expect(storage.get(second) == null);
    try testing.expect(storage.getMut(second) == null);
    try testing.expectEqual(@as(u32, 1), storage.get(first).?.tag);
    try testing.expectEqual(@as(u32, 3), storage.get(third).?.tag);
}
