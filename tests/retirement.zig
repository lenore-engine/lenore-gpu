const std = @import("std");
const gpu = @import("lenore-gpu");

const testing = std.testing;

// A resource whose destruction is observable without a device. The counter is
// shared rather than per value because what these tests are about is when a
// destruction happens, not which value it happened to.
var destroyed: usize = 0;

const Tracked = struct {
    id: u32,

    pub fn deinit(self: *Tracked) void {
        destroyed += 1;
        self.* = undefined;
    }
};

const Queue = gpu.Retirement(Tracked);

fn freshQueue(slots: usize) !Queue {
    destroyed = 0;
    return Queue.init(testing.allocator, slots);
}

test "a retired value outlives the ring before it is destroyed" {
    // The property the whole type exists for. A value retired while slot 0 is
    // recording must survive every slot in flight: slot 1 may still be reading
    // it, and only slot 0 coming round again establishes that the work which
    // could have touched it is complete.
    var queue = try freshQueue(2);
    defer queue.deinit(testing.allocator);

    queue.beginFrame(0);
    try queue.retire(testing.allocator, .{ .id = 7 });
    try testing.expectEqual(@as(usize, 0), destroyed);

    // The next frame is a different slot, so nothing of slot 0's is touched.
    queue.beginFrame(1);
    try testing.expectEqual(@as(usize, 0), destroyed);
    try testing.expectEqual(@as(usize, 1), queue.pending());

    // Slot 0 again: its fence has been waited on by now, and this is the first
    // moment the value can go.
    queue.beginFrame(0);
    try testing.expectEqual(@as(usize, 1), destroyed);
    try testing.expectEqual(@as(usize, 0), queue.pending());
}

test "a deeper ring defers by exactly its own depth" {
    // Three slots means two full frames of grace, not two slots' worth. Getting
    // this off by one destroys a resource one frame early, which is a use after
    // free on the device and nothing a host test of the consumer would show.
    var queue = try freshQueue(3);
    defer queue.deinit(testing.allocator);

    queue.beginFrame(0);
    try queue.retire(testing.allocator, .{ .id = 1 });

    queue.beginFrame(1);
    queue.beginFrame(2);
    try testing.expectEqual(@as(usize, 0), destroyed);

    queue.beginFrame(0);
    try testing.expectEqual(@as(usize, 1), destroyed);
}

test "retirement lands in the open slot and not in a fixed one" {
    var queue = try freshQueue(2);
    defer queue.deinit(testing.allocator);

    queue.beginFrame(1);
    try queue.retire(testing.allocator, .{ .id = 3 });

    // Opening slot 0 must not collect what slot 1 is holding. A retirement that
    // always landed in slot zero would be destroyed here, one frame early.
    queue.beginFrame(0);
    try testing.expectEqual(@as(usize, 0), destroyed);

    queue.beginFrame(1);
    try testing.expectEqual(@as(usize, 1), destroyed);
}

test "several values retired in one frame all go together" {
    var queue = try freshQueue(2);
    defer queue.deinit(testing.allocator);

    queue.beginFrame(0);
    for (0..5) |index| try queue.retire(testing.allocator, .{ .id = @intCast(index) });
    try testing.expectEqual(@as(usize, 5), queue.pending());

    queue.beginFrame(1);
    queue.beginFrame(0);
    try testing.expectEqual(@as(usize, 5), destroyed);
    try testing.expectEqual(@as(usize, 0), queue.pending());
}

test "a slot just collected accepts retirements for the frame it opens" {
    // A slot is emptied and then filled again in the same round, so the list
    // reached by `retire` after `beginFrame` has to be the one just cleared and
    // not a stale reference to what was there before.
    var queue = try freshQueue(2);
    defer queue.deinit(testing.allocator);

    queue.beginFrame(0);
    try queue.retire(testing.allocator, .{ .id = 1 });
    queue.beginFrame(1);
    queue.beginFrame(0);
    try testing.expectEqual(@as(usize, 1), destroyed);

    try queue.retire(testing.allocator, .{ .id = 2 });
    try testing.expectEqual(@as(usize, 1), destroyed);
    try testing.expectEqual(@as(usize, 1), queue.pending());
}

test "teardown destroys what is still held" {
    // A queue that still holds something at teardown is the ordinary case, not
    // a fault: anything retired within a ring's depth of the last frame never
    // saw its slot come round. Measured on the device, where a full corpus walk
    // ended with exactly this state on a clean exit.
    var queue = try freshQueue(2);

    queue.beginFrame(0);
    try queue.retire(testing.allocator, .{ .id = 1 });
    try queue.retire(testing.allocator, .{ .id = 2 });

    // Correct only once the device is idle, which is the one drain this type
    // does not remove: at teardown there is no later frame to wait for.
    queue.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), destroyed);
}

test "teardown destroys nothing twice" {
    // The collected value must be gone from its slot and not merely destroyed
    // in place, or teardown runs its destructor a second time.
    var queue = try freshQueue(2);

    queue.beginFrame(0);
    try queue.retire(testing.allocator, .{ .id = 1 });
    queue.beginFrame(1);
    queue.beginFrame(0);
    try testing.expectEqual(@as(usize, 1), destroyed);

    queue.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), destroyed);
}

test "a ring of no slots is refused" {
    // Zero slots would accept a retirement into nothing and destroy it never,
    // or index out of range. Refused where the other capacities are.
    try testing.expectError(error.ZeroSlots, Queue.init(testing.allocator, 0));
}
