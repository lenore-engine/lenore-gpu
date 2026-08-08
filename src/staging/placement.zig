const std = @import("std");

// Offsets and sizes keep the width of vk.DeviceSize, which the generated
// bindings declare as u64, so nothing truncates at the Vulkan boundary. Nothing
// else here knows about Vulkan: this is the placement arithmetic of the staging
// block pool, and keeping it free of the graphics API is what lets the policy be
// exercised without a device.
const ByteOffset = u64;
const ByteSize = u64;

pub const PolicyError = error{
    // One granule does not fit an empty block, so no sequence of flushes ever
    // satisfies the request. This is the case that must never be answered by
    // retrying, and it is separate for exactly that reason.
    LargerThanBlock,
    InvalidAlignment,
    InvalidGranularity,
    InvalidSize,
};

// Where a chunk goes and how much of the request it takes. The size is what the
// pool can give and never more than what was asked: a caller copies that many
// bytes and asks again for the rest.
pub const Placement = struct {
    block: u32,
    offset: ByteOffset,
    size: ByteSize,
};

pub const Request = struct {
    size: ByteSize,

    // Alignment belongs to the copy rather than to the pool. vkCmdCopyBuffer
    // constrains neither offset, while vkCmdCopyBufferToImage requires
    // bufferOffset to be a multiple of 4 and of the texel block size, and
    // VkPhysicalDeviceLimits::optimalBufferCopyOffsetAlignment is the driver's
    // preference on top of that.
    alignment: ByteSize,

    // The smallest run the caller can copy on its own. A chunk shorter than the
    // whole request is always a whole number of granules, so a vertex stream
    // splits between vertices and an image between rows of texel blocks.
    //
    // A caller that cannot split at all passes its own size, which turns a
    // request too large for a block into LargerThanBlock instead of a chunk it
    // cannot use.
    granularity: ByteSize = 1,
};

pub const Decision = union(enum) {
    // Take this much, here.
    place: Placement,
    // Nothing live has room and the pool may still add a block.
    grow,
    // Nothing live has room and the pool is at its ceiling. The caller submits
    // what it recorded, waits for it, and recycles every block.
    flush,
};

// Grow first, stall last. The buffers, the submission and the fence are the
// caller's; what block a chunk lands in, when to add a block and when to stall
// is decided here.
//
// It is a pure function of the fill levels, so the expensive case reaches a test
// without a device. That matters more than it looks: the stall is the branch
// that costs a load its pipelining, and it is the branch a device test is least
// likely to reach, because it needs a scene large enough to exhaust the pool.
//
// Godot decides the same three actions in
// RenderingDevice::_staging_buffer_allocate and keys block reuse on the frame
// that last wrote a block. Ours are one-shot submissions outside any frame, so
// the key is the submission instead and the pool retires as a whole. Same
// structure, different key.
pub const BlockPolicy = struct {
    block_capacity: ByteSize,
    max_blocks: u32,

    // Chooses where the next chunk of a request goes.
    //
    // `used` is the fill level of every live block, in creation order, and
    // `current` is the block the last placement landed in. The scan starts there
    // rather than at zero: blocks before it are full or hold a tail too short to
    // be worth another copy region, and revisiting them turns one stream into
    // one region per block boundary.
    pub fn choose(
        self: BlockPolicy,
        used: []const ByteSize,
        current: u32,
        request: Request,
    ) PolicyError!Decision {
        if (request.size == 0) return error.InvalidSize;
        if (request.granularity == 0) return error.InvalidGranularity;
        // The zero test comes first because it has to. std/math.zig,
        // isPowerOfTwo asserts a positive argument and then computes
        // int & (int - 1); with safety off the assert is gone and zero reports
        // itself a power of two.
        if (request.alignment == 0 or !std.math.isPowerOfTwo(request.alignment))
            return error.InvalidAlignment;
        // An alignment wider than a block cannot describe a placement in one.
        // Rejecting it is also what keeps alignForward below from overflowing,
        // whose own assert is compiled out in the shipping build: a fill level
        // never exceeds the capacity, so the aligned offset stays under twice it.
        if (request.alignment > self.block_capacity) return error.InvalidAlignment;
        // An empty block starts at offset zero whatever the alignment, so a
        // granule wider than a block never fits, and a caller retrying after a
        // flush would retry forever.
        if (request.granularity > self.block_capacity) return error.LargerThanBlock;

        var block = current;
        while (block < used.len) : (block += 1) {
            if (self.fit(used[block], request)) |placement| {
                return .{ .place = .{
                    .block = block,
                    .offset = placement.offset,
                    .size = placement.size,
                } };
            }
        }

        return if (used.len < self.max_blocks) .grow else .flush;
    }

    const Fit = struct {
        offset: ByteOffset,
        size: ByteSize,
    };

    // What one block can take of the request, or nothing when its remainder is
    // shorter than a granule.
    fn fit(self: BlockPolicy, fill: ByteSize, request: Request) ?Fit {
        const offset = std.mem.alignForward(ByteOffset, fill, request.alignment);
        if (offset >= self.block_capacity) return null;
        const available = self.block_capacity - offset;

        // The whole request when it fits, and otherwise whole granules only.
        // Written this way round so a request that is not a multiple of its
        // granularity still places its remainder.
        const size = if (available >= request.size)
            request.size
        else
            available / request.granularity * request.granularity;
        if (size == 0) return null;
        return .{ .offset = offset, .size = size };
    }
};
