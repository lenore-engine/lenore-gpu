const std = @import("std");
const gpu = @import("lenore-gpu");

const testing = std.testing;

// Stands in for a shared asset: it has the deinit shape the cache requires and
// records its own destruction, so teardown is observed rather than inferred.
const Tracked = struct {
    tag: u32,
    destroyed: *u32,

    pub fn deinit(self: *Tracked) void {
        self.destroyed.* += 1;
        self.* = undefined;
    }
};

const Cache = gpu.RefCache(Tracked);

test "a value lives until its last reference is released" {
    var destroyed: u32 = 0;
    var cache: Cache = .empty;

    try testing.expect(cache.acquire("mesh") == null);
    const inserted = try cache.insert(testing.allocator, "mesh", .{
        .tag = 1,
        .destroyed = &destroyed,
    });
    try testing.expectEqual(@as(u32, 1), inserted.tag);
    try testing.expectEqual(@as(u32, 1), cache.references("mesh"));

    // The second user gets the same value, not a copy.
    const shared = cache.acquire("mesh").?;
    try testing.expectEqual(inserted, shared);
    try testing.expectEqual(@as(u32, 2), cache.references("mesh"));

    cache.release(testing.allocator, "mesh");
    try testing.expectEqual(@as(u32, 0), destroyed);
    try testing.expect(cache.contains("mesh"));

    cache.release(testing.allocator, "mesh");
    try testing.expectEqual(@as(u32, 1), destroyed);
    try testing.expect(!cache.contains("mesh"));
    try testing.expectEqual(@as(u32, 0), cache.count());

    try testing.expectEqual(gpu.RefCacheDeinitStatus.ok, cache.deinit(testing.allocator));
}

test "the cache owns its keys" {
    var destroyed: u32 = 0;
    var cache: Cache = .empty;

    var key: [8]u8 = "texture0".*;
    _ = try cache.insert(testing.allocator, &key, .{ .tag = 1, .destroyed = &destroyed });
    @memset(&key, 0);

    try testing.expect(cache.contains("texture0"));
    try testing.expectEqual(@as(u32, 1), cache.acquire("texture0").?.tag);

    _ = cache.deinit(testing.allocator);
}

test "a duplicate key is rejected and the offered value destroyed" {
    var destroyed: u32 = 0;
    var cache: Cache = .empty;

    const first = try cache.insert(testing.allocator, "shared", .{
        .tag = 1,
        .destroyed = &destroyed,
    });
    try testing.expectError(error.KeyAlreadyPresent, cache.insert(testing.allocator, "shared", .{
        .tag = 2,
        .destroyed = &destroyed,
    }));

    try testing.expectEqual(@as(u32, 1), destroyed);
    try testing.expectEqual(@as(u32, 1), first.tag);
    try testing.expectEqual(@as(u32, 1), cache.references("shared"));
    try testing.expectEqual(@as(u32, 1), cache.count());

    _ = cache.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 2), destroyed);
}

test "a pinned value survives zero references and absorbs extra releases" {
    var destroyed: u32 = 0;
    var cache: Cache = .empty;

    _ = try cache.insert(testing.allocator, "atlas", .{ .tag = 1, .destroyed = &destroyed });
    cache.pin("atlas");
    cache.release(testing.allocator, "atlas");

    try testing.expectEqual(@as(u32, 0), destroyed);
    try testing.expect(cache.contains("atlas"));
    try testing.expectEqual(@as(u32, 0), cache.references("atlas"));

    // A release past zero must not wrap the counter, which would strand the
    // value at a huge count and destroy it never.
    cache.release(testing.allocator, "atlas");
    try testing.expectEqual(@as(u32, 0), cache.references("atlas"));
    try testing.expect(cache.contains("atlas"));

    // Pinned means resident, not immortal: teardown still destroys it, and a
    // pinned value at zero references is not a leak.
    try testing.expectEqual(gpu.RefCacheDeinitStatus.ok, cache.deinit(testing.allocator));
    try testing.expectEqual(@as(u32, 1), destroyed);
}

test "teardown with references outstanding reports a leak" {
    var destroyed: u32 = 0;
    var cache: Cache = .empty;

    _ = try cache.insert(testing.allocator, "held", .{ .tag = 1, .destroyed = &destroyed });
    try testing.expectEqual(gpu.RefCacheDeinitStatus.leak, cache.deinit(testing.allocator));
    try testing.expectEqual(@as(u32, 1), destroyed);
}

test "an unknown key is inert" {
    var cache: Cache = .empty;

    cache.pin("absent");
    cache.release(testing.allocator, "absent");
    try testing.expect(cache.acquire("absent") == null);
    try testing.expectEqual(@as(u32, 0), cache.references("absent"));
    try testing.expect(!cache.contains("absent"));

    try testing.expectEqual(gpu.RefCacheDeinitStatus.ok, cache.deinit(testing.allocator));
}

// Insert allocates twice, for the table and for the owned key, and the value is
// already the cache's by then. Both failures have to leave no entry behind and
// no value alive.
test "a failed insert leaves no entry and destroys the value" {
    var destroyed: u32 = 0;
    var failing = testing.FailingAllocator.init(testing.allocator, .{});
    const allocator = failing.allocator();

    var cache: Cache = .empty;
    _ = try cache.insert(allocator, "first", .{ .tag = 0, .destroyed = &destroyed });

    failing.fail_index = failing.alloc_index;
    failing.resize_fail_index = failing.resize_index;

    var name: [16]u8 = undefined;
    var tag: u32 = 1;
    const failure = while (tag < 4096) : (tag += 1) {
        const key = try std.fmt.bufPrint(&name, "asset{d}", .{tag});
        _ = cache.insert(allocator, key, .{
            .tag = tag,
            .destroyed = &destroyed,
        }) catch |err| break err;
    } else error.NoFailureInduced;
    try testing.expectEqual(error.OutOfMemory, failure);
    try testing.expectEqual(@as(u32, 1), destroyed);

    const key = try std.fmt.bufPrint(&name, "asset{d}", .{tag});
    try testing.expect(!cache.contains(key));
    try testing.expectEqual(tag, cache.count());

    _ = cache.deinit(allocator);
    try testing.expectEqual(tag + 1, destroyed);
    try testing.expect(failing.has_induced_failure);
    try testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
}
