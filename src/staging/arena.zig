const std = @import("std");
const vk = @import("vulkan");
const buffer_module = @import("../buffer.zig");
const placement = @import("staging-placement");
const Context = @import("../context.zig").Context;
const memory = @import("../memory/allocator.zig");

const Buffer = buffer_module.Buffer;

pub const InitError = buffer_module.InitError || error{BufferNotHostVisible};
pub const ReserveError = placement.ReserveError;

// One reserved range: where to write on the host, and where it lives in the
// arena's buffer. The offset feeds Buffer.CopyRegion.source_offset.
pub const Reservation = struct {
    bytes: []u8,
    offset: vk.DeviceSize,
};

// Host-visible transfer source for batched uploads. One buffer, persistently
// mapped by the memory allocator, handed out by a bump pointer and reclaimed as
// a whole.
//
// It is not a fence-tracked ring. Nothing here knows which reservation a given
// submission still reads, so reset frees everything at once and the caller
// carries the invariant. Circular reuse is the caller's loop: reserve until
// OutOfSpace, submit, wait, reset, continue. A ring that reclaims a tail while
// its head is still in flight needs per-segment fences and a consumer that wants
// them.
//
// The placement arithmetic lives in a Vulkan-free module so it can be tested
// without a device. Suballocator is not reused for it: that one tracks and
// releases individual ranges, which is what a general allocator needs and what
// an arena exists to avoid.
pub const StagingArena = struct {
    buffer: Buffer,
    mapped: []u8,
    bump: placement.Bump,

    // The arena is a transfer source in upload memory, which the allocator
    // persistently maps. A class whose memory is not mapped is rejected rather
    // than silently reduced to a buffer nobody can write to.
    pub fn init(
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        size: vk.DeviceSize,
    ) InitError!StagingArena {
        var buffer = try Buffer.init(
            context,
            memory_allocator,
            size,
            .{ .transfer_src_bit = true },
            .upload,
        );
        errdefer buffer.deinit();

        // The allocation can be larger than the buffer, so the arena bounds
        // itself by the size it asked for rather than by the mapped length.
        const mapped = buffer.allocation.mappedBytes() orelse
            return error.BufferNotHostVisible;
        return .{
            .buffer = buffer,
            .mapped = mapped[0..@intCast(buffer.size)],
            .bump = .init(buffer.size),
        };
    }

    // Vulkan specification, vkDestroyBuffer: every submitted copy reading this
    // arena must have completed.
    pub fn deinit(self: *StagingArena) void {
        self.buffer.deinit();
        self.* = undefined;
    }

    // The transfer source for Buffer.recordCopyFrom.
    pub fn source(self: *const StagingArena) *const Buffer {
        return &self.buffer;
    }

    pub fn capacity(self: *const StagingArena) vk.DeviceSize {
        return self.bump.capacity;
    }

    pub fn remaining(self: *const StagingArena) vk.DeviceSize {
        return self.bump.remaining();
    }

    // Reserves size bytes at the given alignment. The returned slice is host
    // memory to write into; the returned offset names the same bytes inside the
    // arena's buffer.
    //
    // Alignment is the caller's because it depends on the copy. vkCmdCopyBuffer
    // constrains neither offset, while vkCmdCopyBufferToImage requires
    // bufferOffset to be a multiple of 4 and of the texel block size, and
    // VkPhysicalDeviceLimits::optimalBufferCopyOffsetAlignment is the driver's
    // preference on top of that.
    pub fn reserve(
        self: *StagingArena,
        size: vk.DeviceSize,
        alignment: vk.DeviceSize,
    ) ReserveError!Reservation {
        const offset = try self.bump.reserve(size, alignment);
        return .{
            .bytes = self.mapped[@intCast(offset)..][0..@intCast(size)],
            .offset = offset,
        };
    }

    // Makes the whole arena available again. The caller guarantees that every
    // submission reading it has completed: this type tracks no fence and cannot
    // check.
    pub fn reset(self: *StagingArena) void {
        self.bump.reset();
    }
};
