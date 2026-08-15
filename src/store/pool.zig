const std = @import("std");

const Allocator = std.mem.Allocator;

// A handle packs a slot index and a generation counter into one u64. The split
// is pinned by the assert below: widening one narrows the other.
const Index = u32;
// Reusing one slot often enough returns its generation to a value an outstanding
// handle still holds, and that handle then reads the current occupant as if it
// were the original. The width is what puts that out of reach: the counter is
// per slot rather than shared, so the wrap needs 2^32 releases of the same slot,
// which is fifty days of releasing one a millisecond.
//
// Half a handle spent on the counter is the split Godot reaches for the same
// reason. An RID is a 64-bit id carrying a 32-bit index and a validator in the
// high half, core/templates/rid.h and core/templates/rid_owner.h.
const Generation = u32;

comptime {
    std.debug.assert(@bitSizeOf(Index) + @bitSizeOf(Generation) == @bitSizeOf(u64));
}

// Indices cover 0..maxInt(Index), so the slot array holds one more entry than
// the largest index.
const max_slots: usize = @as(usize, std.math.maxInt(Index)) + 1;

// Zero is excluded from live generations, as it is from allocator identities, so
// a zero-initialized handle names no live entry.
const invalid_generation: Generation = 0;
const first_generation: Generation = invalid_generation + 1;

pub const AddError = error{PoolExhausted} || Allocator.Error;

// Slot storage with generational handles. The pool owns slot identity and
// lifetime, never the destruction of T: remove returns the evicted value so the
// owning system runs whatever teardown a GPU resource needs. That keeps
// "who destroys the resource" a property of the system, not of the container.
//
// Distinct instantiations produce distinct Handle types, so a mesh handle and a
// texture handle cannot be confused at a call site.
pub fn ResourcePool(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Handle = enum(u64) {
            // Generation zero is never live, so the zero handle resolves to null
            // whatever slot count the pool has.
            invalid = 0,
            _,

            fn index(self: Handle) Index {
                return @truncate(@intFromEnum(self));
            }

            fn generation(self: Handle) Generation {
                return @intCast(@intFromEnum(self) >> @bitSizeOf(Index));
            }

            fn pack(idx: Index, gen: Generation) Handle {
                return @enumFromInt((@as(u64, gen) << @bitSizeOf(Index)) | @as(u64, idx));
            }
        };

        // Which slot a live handle names, or null if it names none.
        //
        // For a system that keeps a fixed array parallel to the slots and wants
        // to index it by the same number the pool chose. A registry holding one
        // descriptor set per slot is the case: the set is allocated once and
        // stays attached to the slot, so the alternative is a second free list
        // tracking the same reuse this one already tracks.
        //
        // Valid only while the handle is live. The number is reused after a
        // remove, which is what the generation in the handle exists to detect,
        // so it is resolved through the same liveness check as `get`.
        pub fn slotIndex(self: *const Self, handle: Handle) ?usize {
            _ = liveSlot(self.slots.items, handle) orelse return null;
            return handle.index();
        }

        const Slot = struct {
            value: T,
            generation: Generation,
            alive: bool,
        };

        slots: std.ArrayList(Slot),
        // Indices of dead slots ready for reuse. Its capacity is grown to match
        // the slot count on every new-slot append, so at most one entry per slot
        // can be queued and remove never allocates.
        free: std.ArrayList(Index),
        live: u32,

        pub const empty: Self = .{ .slots = .empty, .free = .empty, .live = 0 };

        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.slots.deinit(allocator);
            self.free.deinit(allocator);
            self.* = undefined;
        }

        // Inserts value and returns a handle to it, reusing a freed slot when one
        // is available and keeping the generation that slot's release bumped.
        pub fn add(self: *Self, allocator: Allocator, value: T) AddError!Handle {
            if (self.free.pop()) |idx| {
                const slot = &self.slots.items[idx];
                slot.value = value;
                slot.alive = true;
                self.live += 1;
                return Handle.pack(idx, slot.generation);
            }

            if (self.slots.items.len == max_slots) return error.PoolExhausted;
            const new_len = self.slots.items.len + 1;

            // Reserve both arrays before publishing the live slot. If either
            // reservation fails the pool is logically unchanged, so rollback
            // never has to discover an uncounted live entry.
            try self.slots.ensureUnusedCapacity(allocator, 1);
            try self.free.ensureTotalCapacity(allocator, new_len);

            const idx: Index = @intCast(self.slots.items.len);
            self.slots.appendAssumeCapacity(.{
                .value = value,
                .generation = first_generation,
                .alive = true,
            });
            self.live += 1;
            return Handle.pack(idx, first_generation);
        }

        // Resolves a handle to a live entry, or null if it was removed or never
        // existed.
        //
        // The pointer does not survive the next add: the slots are a
        // std.ArrayList and growing one reallocates. Hold the handle.
        pub fn get(self: *const Self, handle: Handle) ?*const T {
            const slot = liveSlot(self.slots.items, handle) orelse return null;
            return &slot.value;
        }

        // Mutable lookup carries the same lifetime rule as get.
        pub fn getMut(self: *Self, handle: Handle) ?*T {
            const slot = liveSlot(self.slots.items, handle) orelse return null;
            return &slot.value;
        }

        // Removes a live entry and returns its value for the caller to destroy.
        // Returns null on a stale or invalid handle, so a double remove is a
        // no-op rather than a second eviction. The slot's generation is bumped,
        // which invalidates every outstanding handle to it.
        pub fn remove(self: *Self, handle: Handle) ?T {
            const slot = liveSlot(self.slots.items, handle) orelse return null;
            const value = slot.value;
            slot.value = undefined;
            slot.alive = false;
            slot.generation = bumped(slot.generation);
            // Capacity for one index per slot was reserved in add.
            self.free.appendAssumeCapacity(handle.index());
            self.live -= 1;
            return value;
        }

        fn bumped(generation: Generation) Generation {
            const next = generation +% 1;
            return if (next == invalid_generation) first_generation else next;
        }

        fn liveSlot(slots: []Slot, handle: Handle) ?*Slot {
            const idx = handle.index();
            if (idx >= slots.len) return null;
            const slot = &slots[idx];
            if (!slot.alive or slot.generation != handle.generation()) return null;
            return slot;
        }

        pub const Iterator = struct {
            slots: []Slot,
            index: usize = 0,

            pub fn next(self: *Iterator) ?*T {
                while (self.index < self.slots.len) {
                    const slot = &self.slots[self.index];
                    self.index += 1;
                    if (slot.alive) return &slot.value;
                }
                return null;
            }
        };

        // Iterates live values in slot order. Neither add nor remove may run
        // during iteration: either can reallocate or recycle slots under the
        // cursor.
        pub fn iterator(self: *Self) Iterator {
            return .{ .slots = self.slots.items };
        }

        pub fn count(self: *const Self) u32 {
            return self.live;
        }

        // Drops every entry and keeps the backing capacity. This invalidates
        // every outstanding handle without bumping any generation, so a handle
        // taken before the call can name a later entry. It is for a full reset
        // where no handle survives, such as dropping a whole scene. The owning
        // system must already have run the teardown the evicted values needed.
        pub fn clearRetainingCapacity(self: *Self) void {
            self.slots.clearRetainingCapacity();
            self.free.clearRetainingCapacity();
            self.live = 0;
        }
    };
}
