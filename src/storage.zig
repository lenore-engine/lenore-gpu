const std = @import("std");
const pool = @import("pool.zig");

const ResourcePool = pool.ResourcePool;

const Allocator = std.mem.Allocator;

// Slot storage that owns the destruction of what it holds. ResourcePool keeps
// slot identity and hands an evicted value back to its caller; this layer is
// where a system says "these values are mine, destroy them with me".
//
// T must expose deinit(*T) void, and it must not need an allocator or a context
// argument: everything a GPU resource needs to destroy itself is already inside
// it. Buffer.deinit has that shape.
//
// A value handed to add is owned from that moment, including on failure, so a
// caller never has to unwind a resource it has already given away.
pub fn OwningStorage(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Pool = ResourcePool(T);
        pub const Handle = Pool.Handle;
        pub const AddError = pool.AddError;

        // Reaching into the pool to remove a value without destroying it drops
        // ownership on the floor. The language has nothing that prevents it, so
        // this comment is the only thing between a caller and that mistake: go
        // through the methods below.
        pool: Pool,

        pub const empty: Self = .{ .pool = .empty };

        // Destroys every live value in slot order, then the slot storage. The
        // caller guarantees that submitted GPU work using these resources has
        // completed, exactly as each T requires.
        pub fn deinit(self: *Self, allocator: Allocator) void {
            var live = self.pool.iterator();
            while (live.next()) |value| value.deinit();
            self.pool.deinit(allocator);
            self.* = undefined;
        }

        // Takes ownership of value and returns its handle. Destroys value if the
        // slot cannot be reserved, so a failed insert leaks nothing.
        pub fn add(self: *Self, allocator: Allocator, value: T) AddError!Handle {
            var owned = value;
            errdefer owned.deinit();
            return self.pool.add(allocator, owned);
        }

        // Destroys the value behind handle and frees its slot. A stale or
        // already removed handle is a no-op, so double release is harmless.
        pub fn remove(self: *Self, handle: Handle) void {
            if (self.pool.remove(handle)) |value| {
                var owned = value;
                owned.deinit();
            }
        }

        // Null once the value is gone, which is what makes a stale handle a
        // caught bug rather than an alias. The pointer is invalidated by the
        // next add. Hold the handle, not the pointer.
        pub fn get(self: *const Self, handle: Handle) ?*const T {
            return self.pool.get(handle);
        }

        pub fn getMut(self: *Self, handle: Handle) ?*T {
            return self.pool.getMut(handle);
        }

        pub fn iterator(self: *Self) Pool.Iterator {
            return self.pool.iterator();
        }

        pub fn count(self: *const Self) u32 {
            return self.pool.count();
        }
    };
}
