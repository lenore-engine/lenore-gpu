const std = @import("std");
const gpu = @import("lenore-gpu");

const testing = std.testing;

// A regression check, not a proof. It asserts one property: none of these types
// has a field that is directly a host allocator.
//
// That is narrower than "per-frame code allocates nothing". It does not see an
// allocator nested inside another field, reached through a pointer, or imported
// as a global, and it cannot see one used inside a method body. Allocation
// freedom rests on review of those methods; this only catches the cheapest way
// to lose it.
const frame_types = [_]type{
    gpu.SkeletonPose,
    gpu.NodeAnimator,
    gpu.Animation,
    gpu.AnimationChannel,
    gpu.StagingArena,
    gpu.MaterialStorage,
    gpu.Mesh,
};

fn holdsHostAllocator(comptime T: type) bool {
    inline for (@typeInfo(T).@"struct".fields) |field| {
        if (field.type == std.mem.Allocator) return true;
    }
    return false;
}

test "no per-frame type has a direct allocator field" {
    inline for (frame_types) |T| {
        if (comptime holdsHostAllocator(T)) {
            std.debug.print("{s} gained an allocator field\n", .{@typeName(T)});
            return error.PerFrameTypeAllocates;
        }
    }
}

// The templates and caches a frame reads from are built once and may hold
// whatever they need, so they are deliberately not in the list above. Naming
// them here says the omission is a decision.
test "the load-time types are outside that rule" {
    try testing.expect(!holdsHostAllocator(gpu.SkeletonTemplate));
    try testing.expect(!holdsHostAllocator(gpu.NodeTemplate));
    // The texture cache does store one, because acquiring a texture is load-time
    // work that allocates a key and a map entry.
    try testing.expect(holdsHostAllocator(gpu.TextureCache));
}

// The staging arena is the one per-frame path that hands out memory, and it does
// so from a fixed reservation rather than from an allocator. Exhausting it is an
// error, never a growth.
test "the arena reports exhaustion instead of growing" {
    const Bump = @import("staging-placement").Bump;

    var bump: Bump = .init(64);
    _ = try bump.reserve(64, 1);
    try testing.expectError(error.OutOfSpace, bump.reserve(1, 1));
    try testing.expectError(error.LargerThanCapacity, bump.reserve(128, 1));

    bump.reset();
    try testing.expectEqual(@as(u64, 0), try bump.reserve(64, 1));
}
