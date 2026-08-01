const std = @import("std");

// Offsets and sizes keep the width of vk.DeviceSize, which the generated
// bindings declare as u64, so nothing truncates at the Vulkan boundary. Nothing
// else here knows about Vulkan: this is the arithmetic of a bump allocator, and
// keeping it free of the graphics API is what lets it be tested without a
// device.
const ByteOffset = u64;
const ByteSize = u64;

pub const ReserveError = error{
    // The request fits the region but not what is left of it. The caller flushes
    // whatever consumed the earlier reservations, resets, and asks again.
    OutOfSpace,
    // The request cannot fit the region at any fill level. Retrying after a
    // reset would loop forever.
    LargerThanCapacity,
    InvalidAlignment,
    InvalidSize,
};

// A bump pointer over a fixed byte region, reclaimed as a whole.
//
// The two exhaustion cases are separate because the caller's recovery differs:
// one is answered by flushing and resetting, the other never is. Collapsing them
// into a single error turns a retry loop into an infinite one.
pub const Bump = struct {
    capacity: ByteSize,
    used: ByteOffset = 0,

    pub fn init(capacity: ByteSize) Bump {
        return .{ .capacity = capacity };
    }

    pub fn remaining(self: *const Bump) ByteSize {
        return self.capacity - self.used;
    }

    // Returns the offset of an aligned run of size bytes and consumes it.
    pub fn reserve(
        self: *Bump,
        size: ByteSize,
        alignment: ByteSize,
    ) ReserveError!ByteOffset {
        if (size == 0) return error.InvalidSize;
        // The zero test comes first because it has to. std/math.zig,
        // isPowerOfTwo asserts a positive argument and then computes
        // int & (int - 1); with safety off the assert is gone and zero reports
        // itself a power of two.
        if (alignment == 0 or !std.math.isPowerOfTwo(alignment))
            return error.InvalidAlignment;
        // An alignment wider than the region cannot describe a placement in it.
        // Rejecting it is also what keeps alignForward below from overflowing,
        // whose own assert is compiled out in the shipping build: used never
        // exceeds the capacity, so the aligned offset stays under twice it.
        if (alignment > self.capacity) return error.InvalidAlignment;

        // An empty region aligns every request to offset zero, so what does not
        // fit the capacity never fits at all.
        if (size > self.capacity) return error.LargerThanCapacity;

        const offset = std.mem.alignForward(ByteOffset, self.used, alignment);
        if (offset > self.capacity or size > self.capacity - offset)
            return error.OutOfSpace;

        self.used = offset + size;
        return offset;
    }

    // Makes the whole region available again. Whether its previous contents are
    // still in use is the caller's to know; nothing here tracks a consumer.
    pub fn reset(self: *Bump) void {
        self.used = 0;
    }
};
