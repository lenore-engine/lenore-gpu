const std = @import("std");
const vk = @import("vulkan");
const buffer_module = @import("../object/buffer.zig");
const placement = @import("staging-placement");
const Context = @import("../device/context.zig").Context;
const memory = @import("../memory/allocator.zig");

const Allocator = std.mem.Allocator;
const Buffer = buffer_module.Buffer;

pub const Request = placement.Request;

// A block must hold one row of the widest image it can be asked to carry,
// because a row is the smallest run an image copy can be split into. The widest
// texel this module accepts is eight bytes: ktx2.zig's support table declares
// r16g16b16a16_sfloat and nothing wider, and the decoded path is four-byte
// RGBA8. A quarter of a megabyte therefore covers a row of 32768 texels, and a
// block-compressed row of the same width is a quarter of that. An image wider
// still is reported as LargerThanBlock rather than copied wrongly.
//
// Godot's block is the same 256 KiB, from
// core/config/project_settings.cpp:1838, and its ceiling is 128 MiB. Ours is
// the smaller 32 MiB because it is what the single arena this replaces held, so
// the change is a capacity fix and not also a memory-budget change.
pub const default_block_capacity: vk.DeviceSize = 256 << 10;
pub const default_max_blocks: u32 = 128;

pub const Config = struct {
    block_capacity: vk.DeviceSize = default_block_capacity,
    max_blocks: u32 = default_max_blocks,
};

// A reservation's bytes are sometimes written as typed elements rather than
// copied byte by byte, and @alignCast on such a slice is undefined behaviour in
// the shipping build rather than a panic. The offset inside a block is aligned
// by the request, but the host pointer is only as aligned as the block's mapped
// base, which comes from the driver's memory requirement. Four bytes is what
// every element this module packs in place needs, and a block that cannot give
// it is refused rather than trusted.
//
// This is not the alignment an image copy asks for. vkCmdCopyBufferToImage
// constrains bufferOffset, which is measured inside the block's buffer and is
// satisfied by the request alone.
pub const host_element_alignment: vk.DeviceSize = 4;

pub const InitError = Allocator.Error || error{InvalidPoolConfig};
pub const GrowError = buffer_module.InitError ||
    error{ BufferNotHostVisible, BlockNotAligned };
pub const ReserveError = placement.PolicyError || GrowError;

// One reserved chunk: where to write it on the host, and where those same bytes
// live on the device. The pair feeds Buffer.CopyRegion.source_offset and
// VkBufferImageCopy::bufferOffset.
pub const Reservation = struct {
    bytes: []u8,
    source: *const Buffer,
    offset: vk.DeviceSize,
};

pub const Outcome = union(enum) {
    reserved: Reservation,
    // Every block is spoken for and the pool is at its ceiling. The caller
    // submits what it has recorded, waits for it, calls recycle, and asks again.
    // Nothing here can submit: the pool owns memory, not work.
    flush_required,
};

const Block = struct {
    buffer: Buffer,
    mapped: []u8,
};

// Host-visible transfer source for uploads: a pool of equally sized blocks,
// created as they are needed and reclaimed together.
//
// It replaces a single bump arena, which had one failure mode that no caller
// could answer. A resource larger than the arena never loaded at all, and a
// batch larger than it could not be split, because there was no point at which
// the arena's contents could be reused. Both are the same missing piece: a
// reservation that can be smaller than the request. Here a chunk is whatever
// fits, the caller copies that much and asks again, and a resource of any size
// is a sequence of chunks rather than a special case.
//
// It tracks no fence. Which submission still reads which block is the caller's
// to know, which is why the exhausted case is reported rather than handled:
// recycle is only correct after the caller has waited. When several submissions
// are in flight at once this becomes a per-block submission tag, and nothing
// here needs one while a single command buffer is recorded at a time.
//
// The block array is allocated once at its ceiling, so a Reservation's source
// pointer stays valid for the pool's life. Growing into a list that reallocates
// would leave every earlier reservation naming a moved buffer.
pub const StagingPool = struct {
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    host_allocator: Allocator,
    policy: placement.BlockPolicy,
    blocks: []Block,
    // Fill level of each live block, parallel to blocks and kept flat because it
    // is what the placement policy reads.
    used: []vk.DeviceSize,
    live: u32,
    current: u32,

    // No device work happens here. A pool that is never asked for a chunk costs
    // two host arrays and no Vulkan object, which is what makes the ceiling
    // affordable to state up front.
    pub fn init(
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        host_allocator: Allocator,
        config: Config,
    ) InitError!StagingPool {
        if (config.block_capacity == 0 or config.max_blocks == 0)
            return error.InvalidPoolConfig;

        const blocks = try host_allocator.alloc(Block, config.max_blocks);
        errdefer host_allocator.free(blocks);
        const used = try host_allocator.alloc(vk.DeviceSize, config.max_blocks);

        return .{
            .context = context,
            .memory_allocator = memory_allocator,
            .host_allocator = host_allocator,
            .policy = .{
                .block_capacity = config.block_capacity,
                .max_blocks = config.max_blocks,
            },
            .blocks = blocks,
            .used = used,
            .live = 0,
            .current = 0,
        };
    }

    // Vulkan specification, vkDestroyBuffer: every submitted copy reading these
    // blocks must have completed.
    pub fn deinit(self: *StagingPool) void {
        for (self.blocks[0..self.live]) |*block| block.buffer.deinit();
        self.host_allocator.free(self.blocks);
        self.host_allocator.free(self.used);
        self.* = undefined;
    }

    // Reserves as much of the request as one block can take, adding a block when
    // the live ones are full and the ceiling allows it.
    //
    // The returned chunk can be shorter than the request. A caller that cannot
    // use a short chunk says so by passing its own size as the request's
    // granularity, and then it is told LargerThanBlock instead.
    //
    // The loop terminates because grow is the only branch that repeats and each
    // pass through it raises live by one, which the policy only asks for while
    // live is below the ceiling.
    pub fn reserve(self: *StagingPool, request: Request) ReserveError!Outcome {
        while (true) {
            const decision = try self.policy.choose(
                self.used[0..self.live],
                self.current,
                request,
            );
            switch (decision) {
                .place => |chosen| {
                    self.used[chosen.block] = chosen.offset + chosen.size;
                    self.current = chosen.block;

                    const block = &self.blocks[chosen.block];
                    return .{ .reserved = .{
                        .bytes = block.mapped[@intCast(chosen.offset)..][0..@intCast(chosen.size)],
                        .source = &block.buffer,
                        .offset = chosen.offset,
                    } };
                },
                .grow => try self.addBlock(),
                .flush => return .flush_required,
            }
        }
    }

    // Makes every block available again. The caller guarantees that every
    // submission reading them has completed: this type tracks no fence and
    // cannot check.
    pub fn recycle(self: *StagingPool) void {
        @memset(self.used[0..self.live], 0);
        self.current = 0;
    }

    pub fn blockCount(self: *const StagingPool) u32 {
        return self.live;
    }

    // Host-visible memory the pool is holding. It only ever grows during a load,
    // so this is the peak by the time one finishes.
    pub fn residentBytes(self: *const StagingPool) vk.DeviceSize {
        return @as(vk.DeviceSize, self.live) * self.policy.block_capacity;
    }

    pub fn blockCapacity(self: *const StagingPool) vk.DeviceSize {
        return self.policy.block_capacity;
    }

    // The pool is a transfer source in upload memory, which the allocator
    // persistently maps. A class whose memory is not mapped is rejected rather
    // than silently reduced to a buffer nobody can write to.
    fn addBlock(self: *StagingPool) GrowError!void {
        // Only reached through the policy's grow, which requires live to be
        // below max_blocks, and the array is max_blocks long.
        var buffer = try Buffer.init(
            self.context,
            self.memory_allocator,
            self.policy.block_capacity,
            .{ .transfer_src_bit = true },
            .upload,
        );
        errdefer buffer.deinit();

        // The allocation can be larger than the buffer, so a block bounds itself
        // by the size it asked for rather than by the mapped length.
        const mapped = buffer.allocation.mappedBytes() orelse
            return error.BufferNotHostVisible;
        if (!std.mem.isAligned(@intFromPtr(mapped.ptr), host_element_alignment))
            return error.BlockNotAligned;

        self.blocks[self.live] = .{
            .buffer = buffer,
            .mapped = mapped[0..@intCast(buffer.size)],
        };
        self.used[self.live] = 0;
        self.live += 1;
    }
};
