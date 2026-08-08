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

// The caches a frame reads from are built once and may hold whatever they need,
// so they are deliberately not in the list above. Naming one here says the
// omission is a decision.
test "the load-time types are outside that rule" {
    // The texture cache does store one, because acquiring a texture is load-time
    // work that allocates a key and a map entry.
    try testing.expect(holdsHostAllocator(gpu.TextureCache));

    // So does the staging pool, for the block array it allocates once at its
    // ceiling. Uploading is load-time work and this type is not on the per-frame
    // path, which is why it is named here rather than in the list above.
    try testing.expect(holdsHostAllocator(gpu.StagingPool));
}
