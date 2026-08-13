const std = @import("std");

const Buffer = @import("../object/buffer.zig").Buffer;
const Image = @import("../object/image.zig").Image;

const log = std.log.scoped(.vulkan);

const Allocator = std.mem.Allocator;

pub const InitError = error{ZeroSlots} || Allocator.Error;

// Destruction deferred until the frame that could still be reading a resource
// has completed.
//
// Vulkan specification, vkDestroyImage and vkDestroyBuffer among others:
// submitted work using the object must have completed before it is destroyed.
// Destroying inline therefore means the caller has to know that a GPU is
// pipelined and to drain it first, and that requirement spreads to every
// consumer that ever releases anything. This is what holds the requirement in
// one place instead.
//
// One list per frame slot. A resource retired while slot N is recording goes in
// slot N's list and is destroyed when slot N comes round again, which is after
// that slot's fence has been waited on and every submission that could have read
// the resource has completed. The bound is therefore the ring's depth, not a
// timer and not a guess.
//
// Parameterized the way `RefCache` and `OwningStorage` are, and for the same
// reason: T owns its own destruction through pub fn deinit(*T) void. It also
// means the whole of this file is reachable from a host test, since what a slot
// does to a value is T's business and not a device's.
pub fn Retirement(comptime T: type) type {
    return struct {
        const Self = @This();

        const Slot = std.ArrayList(T);

        slots: []Slot,
        // The slot a retirement lands in. Held here rather than passed at every
        // call because the point of the type is that a consumer releasing a
        // texture does not have to know which frame is in flight.
        current: usize,

        pub fn init(allocator: Allocator, slot_count: usize) InitError!Self {
            if (slot_count == 0) return error.ZeroSlots;

            const slots = try allocator.alloc(Slot, slot_count);
            for (slots) |*slot| slot.* = .empty;
            return .{ .slots = slots, .current = 0 };
        }

        // Everything still held is destroyed, so this is only correct once the
        // device is idle. That is the caller's to establish and is the one drain
        // this type does not remove: at teardown there is no later frame for a
        // resource to wait for.
        //
        // Unlike the other lifetime containers here, there is no status to
        // report. A non-empty queue at teardown is the ordinary case and not a
        // fault: anything retired within a ring's depth of the last frame never
        // saw its slot come round, and holding it until now is what this type is
        // for. `RefCache` reports a leak because an outstanding reference means
        // someone else still believes they own the value; nothing here is owned
        // by anyone else.
        pub fn deinit(self: *Self, allocator: Allocator) void {
            for (self.slots) |*slot| {
                for (slot.items) |*value| value.deinit();
                slot.deinit(allocator);
            }
            allocator.free(self.slots);
            self.* = undefined;
        }

        // Opens a frame slot: destroys what that slot was holding and leaves
        // retirement pointed at it.
        //
        // One call rather than a collect and an advance, because a caller that
        // can do one without the other will eventually do exactly that: advance
        // without collecting grows a list nothing empties, and collect without
        // advancing retires into a slot that has already gone by. Which of the
        // two happens first inside here is not load bearing; both name the same
        // slot.
        //
        // The caller has waited on this slot's fence. That is what makes the
        // destruction below safe, and it is the whole of this type's contract.
        // Takes no allocator: the capacity a slot reached is kept rather than
        // returned, so a run settles at the high-water mark of what it retires
        // in one frame and the per-frame path stops allocating entirely.
        pub fn beginFrame(self: *Self, slot: usize) void {
            std.debug.assert(slot < self.slots.len);

            const list = &self.slots[slot];
            for (list.items) |*value| value.deinit();
            list.clearRetainingCapacity();
            self.current = slot;
        }

        // Takes ownership of a value and destroys it a ring's depth later.
        //
        // Fallible because it may allocate, and because the alternatives are
        // both wrong: destroying inline is the hazard this type exists to
        // remove, and dropping the value leaks device memory silently. A caller
        // that cannot proceed without retiring propagates it like any other
        // allocation failure.
        //
        // Cold by construction. A resource released every frame is a resource
        // that should not have been created every frame, and the per-frame path
        // through this type is `beginFrame`, which walks a list and allocates
        // nothing.
        pub fn retire(self: *Self, allocator: Allocator, value: T) Allocator.Error!void {
            try self.slots[self.current].append(allocator, value);
        }

        // What has been retired and not yet collected. Named for the leak
        // report, which is the only thing that can say whether a run gave back
        // what it took.
        pub fn pending(self: *const Self) usize {
            var total: usize = 0;
            for (self.slots) |slot| total += slot.items.len;
            return total;
        }
    };
}

// What this module defers the destruction of. One union rather than a queue per
// type, so that a frame opens one slot and the order things are destroyed in is
// stated in one place: a variant added later whose destruction has to precede
// another's has somewhere to say so, and `Retirement` above stays a container
// that knows nothing about devices.
//
// The whole value moves in. `Image` and `Buffer` each carry the context and the
// allocator their destruction needs, so nothing here has to be handed a device,
// and a variant is complete on its own.
pub const Resource = union(enum) {
    image: Image,
    buffer: Buffer,

    pub fn deinit(self: *Resource) void {
        switch (self.*) {
            inline else => |*owned| owned.deinit(),
        }
    }
};

pub const ResourceRetirement = Retirement(Resource);

// Retires a resource, or destroys it now if the queue will not take it.
//
// The degradation, stated once. Every caller of this is on a teardown path with
// nowhere to report a failure, so the choice is between three answers and only
// this one is defensible: leaking the resource loses device memory silently,
// destroying it without a drain is the hazard this file exists to remove, and
// draining first is merely slow. It costs a stall in a case that needs the host
// to be out of memory.
//
// The device to drain comes from the resource itself, since every variant holds
// the context its own destruction needs.
pub fn retireOrDestroy(
    queue: *ResourceRetirement,
    allocator: Allocator,
    value: Resource,
) void {
    queue.retire(allocator, value) catch {
        log.warn("could not defer a destruction; draining the device instead", .{});
        var owned = value;
        switch (owned) {
            inline else => |*resource| resource.context.waitIdle() catch |err| {
                log.err("device did not drain before destroying a resource: {t}", .{err});
            },
        }
        owned.deinit();
    };
}
