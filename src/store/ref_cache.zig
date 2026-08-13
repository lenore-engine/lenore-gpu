const std = @import("std");

const Allocator = std.mem.Allocator;

pub const InsertError = error{KeyAlreadyPresent} || Allocator.Error;

pub const DeinitStatus = enum {
    ok,
    leak,
};

// Content-addressed, reference-counted store for shared resources.
//
// The key is a content identity: a file path, or a synthetic name for embedded
// data. Loading the same asset twice returns the same T and takes a reference
// instead of duplicating it, and a value is destroyed when its last reference
// goes. An asset used by several live scenes survives until the last one drops
// it; an asset unique to one scene dies with it. A pinned value opts out and
// stays resident past zero references, for what is preloaded once and kept hot.
//
// This is the lifetime layer for shared assets, deliberately separate from
// OwningStorage in owning.zig, which gives handles to uniquely owned resources
// with no sharing at all.
//
// T must expose pub fn deinit(*T) void, the same shape OwningStorage requires.
// The cache destroys what it still holds at teardown; a value whose last
// reference goes is handed back to the caller instead, because when a device
// resource may be destroyed is a question this container cannot answer and the
// caller can. Keys are owned either way, duplicated on insert and freed with the
// entry.
pub fn RefCache(comptime T: type) type {
    return struct {
        const Self = @This();

        const Entry = struct {
            value: T,
            references: u32,
            pinned: bool,
        };

        map: std.StringHashMapUnmanaged(Entry),

        pub const empty: Self = .{ .map = .empty };

        // Destroys every remaining value and frees every owned key. Reports a
        // leak when an entry still holds references: someone tore the cache down
        // while a user of the asset was still alive. Pinned entries at zero
        // references are resident by request and are not leaks.
        pub fn deinit(self: *Self, allocator: Allocator) DeinitStatus {
            var status: DeinitStatus = .ok;
            var entries = self.map.iterator();
            while (entries.next()) |entry| {
                if (entry.value_ptr.references > 0) status = .leak;
                entry.value_ptr.value.deinit();
                allocator.free(entry.key_ptr.*);
            }
            self.map.deinit(allocator);
            self.* = undefined;
            return status;
        }

        // Takes a reference to a present value and returns it, or null when the
        // key is absent, which is the caller's signal to create the value and
        // hand it to insert.
        //
        // The pointer does not survive the next insert: std/hash_map.zig grows
        // through ensureTotalCapacity, which reallocates the entries. Anything
        // that must outlive an insert copies the value out.
        pub fn acquire(self: *Self, key: []const u8) ?*T {
            const entry = self.map.getPtr(key) orelse return null;
            entry.references += 1;
            return &entry.value;
        }

        // Registers a freshly created value at one reference and returns it. The
        // cache owns the value from this call, including on failure, so a caller
        // never has to unwind a resource it has already handed over.
        //
        // A key that is already present is an error rather than an assert: keys
        // come from asset data, and the miss-then-insert sequence is the
        // caller's to get right, not something a release build should trust
        // silently.
        pub fn insert(
            self: *Self,
            allocator: Allocator,
            key: []const u8,
            value: T,
        ) InsertError!*T {
            var owned = value;
            errdefer owned.deinit();

            const entry = try self.map.getOrPut(allocator, key);
            if (entry.found_existing) return error.KeyAlreadyPresent;
            errdefer self.map.removeByPtr(entry.key_ptr);

            // std/hash_map.zig, getOrPutContext: on a miss it writes the key it
            // was given into the new entry. The duplicate below has the same
            // bytes, so replacing it in place leaves the hash valid.
            entry.key_ptr.* = try allocator.dupe(u8, key);
            entry.value_ptr.* = .{ .value = owned, .references = 1, .pinned = false };
            return &entry.value_ptr.value;
        }

        // Drops one reference. Returns the value, moved out and no longer known
        // to the cache, when the last reference goes and it is not pinned, and
        // null in every other case: still referenced, pinned, or an unknown key.
        // An unknown key is a no-op because a neutral fallback that never
        // entered the cache is released like anything else.
        //
        // The value comes back rather than being destroyed here so that a
        // caller holding a GPU resource can retire it against the frame ring
        // instead. Destroying it inline would be correct only with the device
        // idle, and this container is in no position to know that.
        //
        // Ignoring the result leaks whatever it holds, which is why it is a
        // value and not an out parameter: Zig refuses a call statement that
        // discards one, so every call site has to say what it does with it.
        pub fn release(self: *Self, allocator: Allocator, key: []const u8) ?T {
            const entry = self.map.getEntry(key) orelse return null;
            // Only a pinned entry can sit at zero references. Releasing one
            // again would wrap the counter and strand the value forever, so the
            // extra release is dropped instead.
            if (entry.value_ptr.references == 0) return null;

            entry.value_ptr.references -= 1;
            if (entry.value_ptr.references > 0 or entry.value_ptr.pinned) return null;

            const value = entry.value_ptr.value;
            const owned_key = entry.key_ptr.*;
            self.map.removeByPtr(entry.key_ptr);
            allocator.free(owned_key);
            return value;
        }

        // Marks a present value resident: it survives its reference count
        // reaching zero and is destroyed only at deinit. An unknown key is a
        // no-op.
        pub fn pin(self: *Self, key: []const u8) void {
            if (self.map.getPtr(key)) |entry| entry.pinned = true;
        }

        pub fn references(self: *const Self, key: []const u8) u32 {
            const entry = self.map.getPtr(key) orelse return 0;
            return entry.references;
        }

        pub fn contains(self: *const Self, key: []const u8) bool {
            return self.map.contains(key);
        }

        pub fn count(self: *const Self) u32 {
            return self.map.count();
        }
    };
}
