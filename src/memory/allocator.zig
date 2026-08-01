const std = @import("std");
const vk = @import("vulkan");
const Context = @import("../context.zig").Context;
const Suballocator = @import("memory-suballocator").Suballocator;

const Allocator = std.mem.Allocator;

// Zero is excluded from live identities so zero-initialized and stale handles
// cannot accidentally name the first allocator-owned object.
const invalid_identity: u64 = 0;
const first_identity: u64 = invalid_identity + 1;

pub const BufferClass = enum {
    device,
    upload,
    readback,
};

pub const Config = struct {
    device_buffer_block_size: vk.DeviceSize,
    device_image_block_size: vk.DeviceSize,
    upload_buffer_block_size: vk.DeviceSize,
    readback_buffer_block_size: vk.DeviceSize,
};

pub const InitError = error{InvalidBlockSize};
pub const AllocationError = error{
    MemoryBudgetExceeded,
    NoSuitableMemoryType,
    SizeOverflow,
    MapReturnedNull,
} || Allocator.Error ||
    vk.DeviceWrapper.AllocateMemoryError ||
    vk.DeviceWrapper.MapMemoryError;
pub const BufferAllocationError = AllocationError || vk.DeviceWrapper.BindBufferMemoryError;
pub const ImageAllocationError = AllocationError || vk.DeviceWrapper.BindImageMemoryError;
pub const FreeError = error{InvalidAllocation};

pub const DeinitStatus = enum {
    ok,
    leak,
};

pub const Allocation = struct {
    block_index: usize,
    block_generation: u64,
    allocation_id: u64,
    memory_handle: vk.DeviceMemory,
    memory_offset: vk.DeviceSize,
    byte_len: vk.DeviceSize,
    mapped_address: ?[*]u8,
    pub fn memory(self: Allocation) vk.DeviceMemory {
        return self.memory_handle;
    }

    pub fn offset(self: Allocation) vk.DeviceSize {
        return self.memory_offset;
    }

    pub fn size(self: Allocation) vk.DeviceSize {
        return self.byte_len;
    }

    // Vulkan specification, Memory Mapping: the caller synchronizes host access
    // with submitted GPU reads and writes for the lifetime of this mapped slice.
    pub fn mappedBytes(self: Allocation) ?[]u8 {
        const address = self.mapped_address orelse return null;
        return address[0..@intCast(self.byte_len)];
    }
};

const Pool = enum {
    device_buffer,
    device_image,
    upload_buffer,
    readback_buffer,
};

const Block = struct {
    memory: vk.DeviceMemory,
    size: vk.DeviceSize,
    memory_type_index: u32,
    pool: Pool,
    mapped: ?[*]u8,
    suballocator: Suballocator,
    release_when_empty: bool,
};

const BlockSlot = struct {
    generation: u64 = invalid_identity,
    block: ?Block = null,
};

const MemoryTypeChoice = struct {
    index: u32,
    heap_index: u32,
};

// A driver's preference does not override configured block policy. A required
// dedicated allocation names the resource that must own the complete block.
const DedicatedResource = union(enum) {
    none,
    buffer: vk.Buffer,
    image: vk.Image,
};

const Request = struct {
    pool: Pool,
    requirements: vk.MemoryRequirements,
    dedicated: DedicatedResource,
};

pub const MemoryAllocator = struct {
    context: *const Context,
    host_allocator: Allocator,
    io: std.Io,
    config: Config,
    mutex: std.Io.Mutex = .init,
    blocks: std.ArrayList(BlockSlot) = .empty,
    next_allocation_id: u64 = first_identity,

    pub fn init(
        context: *const Context,
        host_allocator: Allocator,
        io: std.Io,
        config: Config,
    ) InitError!MemoryAllocator {
        if (config.device_buffer_block_size == 0 or
            config.device_image_block_size == 0 or
            config.upload_buffer_block_size == 0 or
            config.readback_buffer_block_size == 0)
        {
            return error.InvalidBlockSize;
        }
        return .{
            .context = context,
            .host_allocator = host_allocator,
            .io = io,
            .config = config,
        };
    }

    // Vulkan specification, vkFreeMemory: submitted work using these blocks must
    // be complete before teardown. The caller must also exclude calls already
    // waiting on the mutex; unlocking an object during destruction cannot make a
    // waiting call safe after the object becomes undefined.
    pub fn deinit(self: *MemoryAllocator) DeinitStatus {
        self.mutex.lockUncancelable(self.io);

        var status: DeinitStatus = .ok;
        for (self.blocks.items) |*slot| {
            if (slot.block) |*block| {
                if (block.suballocator.hasAllocations()) status = .leak;
                self.destroyBlock(block);
                slot.block = null;
            }
        }
        self.blocks.deinit(self.host_allocator);
        self.mutex.unlock(self.io);
        self.* = undefined;
        return status;
    }

    // This ordinary-buffer path excludes sparse buffers and buffers created with
    // VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT. The Context does not enable the
    // bufferDeviceAddress feature; adding it also requires allocation-flag pools.
    pub fn allocateBuffer(
        self: *MemoryAllocator,
        buffer: vk.Buffer,
        class: BufferClass,
    ) BufferAllocationError!Allocation {
        var dedicated = vk.MemoryDedicatedRequirements{
            .prefers_dedicated_allocation = undefined,
            .requires_dedicated_allocation = undefined,
        };
        var requirements = vk.MemoryRequirements2{
            .p_next = @ptrCast(&dedicated),
            .memory_requirements = undefined,
        };
        self.context.device.getBufferMemoryRequirements2(
            &.{ .buffer = buffer },
            &requirements,
        );

        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);

        const allocation = try self.allocateLocked(.{
            .pool = switch (class) {
                .device => .device_buffer,
                .upload => .upload_buffer,
                .readback => .readback_buffer,
            },
            .requirements = requirements.memory_requirements,
            .dedicated = if (dedicated.requires_dedicated_allocation == .true)
                .{ .buffer = buffer }
            else
                .none,
        });
        errdefer self.freeLocked(allocation) catch |err| switch (err) {
            error.InvalidAllocation => @panic("allocator rejected its own buffer allocation"),
        };
        try self.context.device.bindBufferMemory(
            buffer,
            allocation.memory_handle,
            allocation.memory_offset,
        );
        return allocation;
    }

    // Vulkan specification, Buffer-Image Granularity: image blocks contain only
    // non-sparse, optimal-tiling images. Buffers use separate blocks and linear
    // images are outside this API, so unlike classes never share a memory block.
    pub fn allocateOptimalImage(
        self: *MemoryAllocator,
        image: vk.Image,
    ) ImageAllocationError!Allocation {
        var dedicated = vk.MemoryDedicatedRequirements{
            .prefers_dedicated_allocation = undefined,
            .requires_dedicated_allocation = undefined,
        };
        var requirements = vk.MemoryRequirements2{
            .p_next = @ptrCast(&dedicated),
            .memory_requirements = undefined,
        };
        self.context.device.getImageMemoryRequirements2(
            &.{ .image = image },
            &requirements,
        );

        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);

        const allocation = try self.allocateLocked(.{
            .pool = .device_image,
            .requirements = requirements.memory_requirements,
            .dedicated = if (dedicated.requires_dedicated_allocation == .true)
                .{ .image = image }
            else
                .none,
        });
        errdefer self.freeLocked(allocation) catch |err| switch (err) {
            error.InvalidAllocation => @panic("allocator rejected its own image allocation"),
        };
        try self.context.device.bindImageMemory(
            image,
            allocation.memory_handle,
            allocation.memory_offset,
        );
        return allocation;
    }

    pub fn free(self: *MemoryAllocator, allocation: Allocation) FreeError!void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.freeLocked(allocation);
    }

    // Blocks currently held. Without it the effect of trim is unobservable, and
    // an allocator that cannot report what it holds cannot be checked against
    // what it should hold.
    pub fn liveBlockCount(self: *MemoryAllocator) usize {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);

        var live: usize = 0;
        for (self.blocks.items) |*slot| {
            if (slot.block != null) live += 1;
        }
        return live;
    }

    pub fn trim(self: *MemoryAllocator) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);

        for (self.blocks.items) |*slot| {
            const block = &(slot.block orelse continue);
            if (block.suballocator.hasAllocations()) continue;
            self.destroyBlock(block);
            slot.block = null;
        }
    }

    fn freeLocked(self: *MemoryAllocator, allocation: Allocation) FreeError!void {
        if (allocation.block_index >= self.blocks.items.len)
            return error.InvalidAllocation;
        const slot = &self.blocks.items[allocation.block_index];
        if (slot.generation != allocation.block_generation)
            return error.InvalidAllocation;
        const block = &(slot.block orelse return error.InvalidAllocation);
        if (block.memory != allocation.memory_handle or
            allocation.byte_len == 0 or
            allocation.memory_offset > block.size or
            allocation.byte_len > block.size - allocation.memory_offset)
        {
            return error.InvalidAllocation;
        }

        block.suballocator.free(
            allocation.allocation_id,
            allocation.memory_offset,
            allocation.byte_len,
        ) catch return error.InvalidAllocation;
        if (block.release_when_empty and !block.suballocator.hasAllocations()) {
            self.destroyBlock(block);
            slot.block = null;
        }
    }

    fn allocateLocked(self: *MemoryAllocator, request: Request) AllocationError!Allocation {
        if (request.requirements.size == 0 or request.requirements.alignment == 0)
            return error.SizeOverflow;

        const allocation_id = self.nextAllocationId();
        if (request.dedicated == .none) {
            for (self.blocks.items, 0..) |*slot, index| {
                const block = &(slot.block orelse continue);
                const type_bit = @as(u32, 1) << @intCast(block.memory_type_index);
                if (block.pool != request.pool or
                    request.requirements.memory_type_bits & type_bit == 0 or
                    block.release_when_empty)
                {
                    continue;
                }
                const offset = try block.suballocator.allocate(
                    self.host_allocator,
                    request.requirements.size,
                    request.requirements.alignment,
                    allocation_id,
                ) orelse continue;
                return makeAllocation(slot.*, block.*, index, offset, request.requirements.size, allocation_id);
            }
        }

        const configured_size = self.blockSize(request.pool);
        const requested_size = request.requirements.size;
        const dedicated = request.dedicated != .none;
        const block_size = if (dedicated)
            requested_size
        else
            @max(configured_size, requested_size);
        const choice = try self.chooseMemoryType(
            request.requirements.memory_type_bits,
            request.pool,
            block_size,
        );
        return self.allocateFromNewBlock(
            request,
            choice,
            block_size,
            dedicated or requested_size > configured_size,
            allocation_id,
        );
    }

    fn allocateFromNewBlock(
        self: *MemoryAllocator,
        request: Request,
        choice: MemoryTypeChoice,
        block_size: vk.DeviceSize,
        release_when_empty: bool,
        allocation_id: u64,
    ) AllocationError!Allocation {
        const dedicated = request.dedicated != .none;
        var dedicated_info = vk.MemoryDedicatedAllocateInfo{};
        switch (request.dedicated) {
            .none => {},
            .buffer => |buffer| dedicated_info.buffer = buffer,
            .image => |image| dedicated_info.image = image,
        }
        const memory = try self.context.device.allocateMemory(&.{
            .p_next = if (dedicated) @ptrCast(&dedicated_info) else null,
            .allocation_size = block_size,
            .memory_type_index = choice.index,
        }, null);
        var owns_memory = true;
        errdefer if (owns_memory) self.context.device.freeMemory(memory, null);

        const host_accessible = request.pool == .upload_buffer or
            request.pool == .readback_buffer;
        const mapped: ?[*]u8 = if (host_accessible) mapped: {
            const address = try self.context.device.mapMemory(
                memory,
                0,
                block_size,
                .{},
            ) orelse return error.MapReturnedNull;
            break :mapped @ptrCast(address);
        } else null;
        errdefer if (owns_memory and mapped != null)
            self.context.device.unmapMemory(memory);

        var suballocator = Suballocator.init(self.host_allocator, block_size) catch |err| switch (err) {
            error.InvalidSize => return error.SizeOverflow,
            else => |other| return other,
        };
        errdefer suballocator.deinit(self.host_allocator);

        const offset = (try suballocator.allocate(
            self.host_allocator,
            request.requirements.size,
            request.requirements.alignment,
            allocation_id,
        )) orelse return error.SizeOverflow;
        const slot_index = try self.acquireSlot();
        const slot = &self.blocks.items[slot_index];
        slot.block = .{
            .memory = memory,
            .size = block_size,
            .memory_type_index = choice.index,
            .pool = request.pool,
            .mapped = mapped,
            .suballocator = suballocator,
            .release_when_empty = release_when_empty,
        };
        owns_memory = false;
        suballocator = undefined;

        return makeAllocation(
            slot.*,
            slot.block.?,
            slot_index,
            offset,
            request.requirements.size,
            allocation_id,
        );
    }

    fn nextAllocationId(self: *MemoryAllocator) u64 {
        const id = self.next_allocation_id;
        self.next_allocation_id +%= 1;
        if (self.next_allocation_id == invalid_identity)
            self.next_allocation_id = first_identity;
        return id;
    }

    fn acquireSlot(self: *MemoryAllocator) Allocator.Error!usize {
        for (self.blocks.items, 0..) |*slot, index| {
            if (slot.block == null) {
                slot.generation +%= 1;
                if (slot.generation == invalid_identity)
                    slot.generation = first_identity;
                return index;
            }
        }
        try self.blocks.append(self.host_allocator, .{
            .generation = first_identity,
        });
        return self.blocks.items.len - 1;
    }

    fn chooseMemoryType(
        self: *const MemoryAllocator,
        type_bits: u32,
        pool: Pool,
        allocation_size: vk.DeviceSize,
    ) (error{ NoSuitableMemoryType, MemoryBudgetExceeded })!MemoryTypeChoice {
        var best: ?MemoryTypeChoice = null;
        var best_is_preferred = false;
        var compatible_type_exists = false;
        for (self.context.memory_properties.memory_types[0..self.context.memory_properties.memory_type_count], 0..) |memory_type, index| {
            if (type_bits & (@as(u32, 1) << @intCast(index)) == 0) continue;
            const flags = memory_type.property_flags;
            const preferred = switch (pool) {
                .device_buffer, .device_image => if (flags.device_local_bit)
                    !flags.host_visible_bit
                else
                    continue,
                .upload_buffer => if (flags.host_visible_bit and flags.host_coherent_bit)
                    flags.device_local_bit
                else
                    continue,
                .readback_buffer => if (flags.host_visible_bit and flags.host_coherent_bit)
                    flags.host_cached_bit
                else
                    continue,
            };
            compatible_type_exists = true;
            if (self.context.memory_budget_enabled and
                !self.fitsMemoryBudget(memory_type.heap_index, allocation_size))
            {
                continue;
            }
            if (best == null or (preferred and !best_is_preferred)) {
                best_is_preferred = preferred;
                best = .{
                    .index = @intCast(index),
                    .heap_index = memory_type.heap_index,
                };
            }
        }
        if (best) |choice| return choice;
        return if (compatible_type_exists)
            error.MemoryBudgetExceeded
        else
            error.NoSuitableMemoryType;
    }

    fn blockSize(self: *const MemoryAllocator, pool: Pool) vk.DeviceSize {
        return switch (pool) {
            .device_buffer => self.config.device_buffer_block_size,
            .device_image => self.config.device_image_block_size,
            .upload_buffer => self.config.upload_buffer_block_size,
            .readback_buffer => self.config.readback_buffer_block_size,
        };
    }

    fn fitsMemoryBudget(
        self: *const MemoryAllocator,
        heap_index: u32,
        amount: vk.DeviceSize,
    ) bool {
        var budget = vk.PhysicalDeviceMemoryBudgetPropertiesEXT{
            .heap_budget = undefined,
            .heap_usage = undefined,
        };
        var properties = vk.PhysicalDeviceMemoryProperties2{
            .p_next = @ptrCast(&budget),
            .memory_properties = undefined,
        };
        self.context.instance.getPhysicalDeviceMemoryProperties2(
            self.context.physical_device,
            &properties,
        );
        const available = budget.heap_budget[heap_index] -| budget.heap_usage[heap_index];
        return amount <= available;
    }

    fn destroyBlock(self: *MemoryAllocator, block: *Block) void {
        block.suballocator.deinit(self.host_allocator);
        if (block.mapped != null) self.context.device.unmapMemory(block.memory);
        self.context.device.freeMemory(block.memory, null);
        block.* = undefined;
    }
};

fn makeAllocation(
    slot: BlockSlot,
    block: Block,
    block_index: usize,
    offset: vk.DeviceSize,
    size: vk.DeviceSize,
    allocation_id: u64,
) Allocation {
    return .{
        .block_index = block_index,
        .block_generation = slot.generation,
        .allocation_id = allocation_id,
        .memory_handle = block.memory,
        .memory_offset = offset,
        .byte_len = size,
        .mapped_address = if (block.mapped) |mapped| mapped + @as(usize, @intCast(offset)) else null,
    };
}
