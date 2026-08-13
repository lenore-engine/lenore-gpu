const std = @import("std");
const vk = @import("vulkan");
const Context = @import("../device/context.zig").Context;
const memory = @import("../memory/allocator.zig");

// Whole-buffer helpers begin at the first byte of each bound resource.
const buffer_start: vk.DeviceSize = 0;

// Context enables no feature-gated buffer usage. Keeping the accepted mask to
// Vulkan 1.0 core bits prevents extension and reserved bits reaching creation.
const supported_usage = vk.BufferUsageFlags{
    .transfer_src_bit = true,
    .transfer_dst_bit = true,
    .uniform_texel_buffer_bit = true,
    .storage_texel_buffer_bit = true,
    .uniform_buffer_bit = true,
    .storage_buffer_bit = true,
    .index_buffer_bit = true,
    .vertex_buffer_bit = true,
    .indirect_buffer_bit = true,
};

pub const InitError = error{
    AllocatorDeviceMismatch,
    EmptyUsage,
    InvalidSize,
    SizeLimitExceeded,
    UnsupportedUsage,
} || vk.DeviceWrapper.CreateBufferError || memory.BufferAllocationError;

pub const UploadError = error{
    BufferNotHostVisible,
    DataOutOfBounds,
};

pub const CopyError = error{
    DestinationRangeOutOfBounds,
    DifferentDevice,
    InvalidSize,
    MissingTransferDestinationUsage,
    MissingTransferSourceUsage,
    SameBuffer,
    SourceRangeOutOfBounds,
};

// One transfer region. Both offsets default to the first byte, so a whole-buffer
// copy stays a one-field literal, and the size is always stated: a staging
// buffer feeding several destinations has no size the destination could imply.
pub const CopyRegion = struct {
    source_offset: vk.DeviceSize = buffer_start,
    destination_offset: vk.DeviceSize = buffer_start,
    size: vk.DeviceSize,
};

// What the checks below need to know about a buffer: no context pointer, so the
// policy can be exercised without a device. This is the same split the staging
// arena makes, and for the same reason: everything here that can be wrong is
// arithmetic and flags, and none of it should need a GPU to test.
pub const Descriptor = struct {
    device: vk.Device,
    handle: vk.Buffer,
    size: vk.DeviceSize,
    usage: vk.BufferUsageFlags,
};

pub const CreateRequest = struct {
    device: vk.Device,
    allocator_device: vk.Device,
    size: vk.DeviceSize,
    max_buffer_size: vk.DeviceSize,
    usage: vk.BufferUsageFlags,
};

// Vulkan specification, VkBufferCreateInfo: a buffer has a non-zero size no
// greater than maxBufferSize, and a usage this module supports.
pub fn validateCreate(request: CreateRequest) InitError!void {
    if (request.size == 0) return error.InvalidSize;
    if (request.size > request.max_buffer_size) return error.SizeLimitExceeded;
    if (request.device != request.allocator_device) return error.AllocatorDeviceMismatch;
    if (std.meta.eql(request.usage, vk.BufferUsageFlags{})) return error.EmptyUsage;
    if (!supported_usage.contains(request.usage)) return error.UnsupportedUsage;
}

// Vulkan specification, vkCmdCopyBuffer: a non-zero size, both ranges inside
// their buffers, the transfer usages present, and one device. The range tests
// are written as subtractions so an offset near the end of a buffer cannot
// overflow into looking valid.
pub fn validateCopy(
    destination: Descriptor,
    source: Descriptor,
    region: CopyRegion,
) CopyError!void {
    if (destination.device != source.device) return error.DifferentDevice;
    if (destination.handle == source.handle) return error.SameBuffer;
    if (!source.usage.transfer_src_bit) return error.MissingTransferSourceUsage;
    if (!destination.usage.transfer_dst_bit) return error.MissingTransferDestinationUsage;
    if (region.size == 0) return error.InvalidSize;
    if (region.source_offset > source.size or
        region.size > source.size - region.source_offset)
        return error.SourceRangeOutOfBounds;
    if (region.destination_offset > destination.size or
        region.size > destination.size - region.destination_offset)
        return error.DestinationRangeOutOfBounds;
}

pub const Buffer = struct {
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    handle: vk.Buffer,
    allocation: memory.Allocation,
    size: vk.DeviceSize,
    usage: vk.BufferUsageFlags,

    pub fn init(
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        size: vk.DeviceSize,
        usage: vk.BufferUsageFlags,
        class: memory.BufferClass,
    ) InitError!Buffer {
        try validateCreate(.{
            .device = context.device.handle,
            .allocator_device = memory_allocator.context.device.handle,
            .size = size,
            .max_buffer_size = context.max_buffer_size,
            .usage = usage,
        });

        const handle = try context.device.createBuffer(&.{
            .size = size,
            .usage = usage,
            .sharing_mode = .exclusive,
        }, null);
        errdefer context.device.destroyBuffer(handle, null);

        const allocation = try memory_allocator.allocateBuffer(handle, class);
        return .{
            .context = context,
            .memory_allocator = memory_allocator,
            .handle = handle,
            .allocation = allocation,
            .size = size,
            .usage = usage,
        };
    }

    // Vulkan specification, vkDestroyBuffer and vkFreeMemory: submitted work
    // using this buffer must complete before either owned object is released.
    pub fn deinit(self: *Buffer) void {
        self.context.device.destroyBuffer(self.handle, null);
        self.memory_allocator.free(self.allocation) catch |err| switch (err) {
            error.InvalidAllocation => @panic("buffer owns an invalid memory allocation"),
        };
        self.* = undefined;
    }

    pub fn upload(self: *Buffer, data: []const u8) UploadError!void {
        return self.uploadAt(buffer_start, data);
    }

    // Vulkan specification, Memory Mapping: the caller synchronizes host writes
    // with submitted GPU accesses to the same byte range.
    pub fn uploadAt(
        self: *Buffer,
        offset: vk.DeviceSize,
        data: []const u8,
    ) UploadError!void {
        const data_size: vk.DeviceSize = @intCast(data.len);
        if (offset > self.size or data_size > self.size - offset)
            return error.DataOutOfBounds;
        const host_bytes = self.allocation.mappedBytes() orelse
            return error.BufferNotHostVisible;
        @memcpy(host_bytes[@intCast(offset)..][0..data.len], data);
    }

    // The buffer's bytes as the host sees them, or null when its memory class is
    // not mapped. Read-only: writes go through upload, which bounds-checks them.
    //
    // Vulkan specification, Memory Mapping: the caller synchronizes this read
    // against submitted GPU writes to the same range. Readback memory is
    // coherent, so a completed submission is enough and no invalidate is needed.
    pub fn describe(self: *const Buffer) Descriptor {
        return .{
            .device = self.context.device.handle,
            .handle = self.handle,
            .size = self.size,
            .usage = self.usage,
        };
    }

    pub fn mapped(self: *const Buffer) ?[]const u8 {
        return self.allocation.mappedBytes();
    }

    // Vulkan specification, vkCmdCopyBuffer: command_buffer must be unprotected,
    // belong to the same device, and be recording outside render and video scopes
    // in a transfer-capable pool. The caller owns queue-family transfers and the
    // memory dependencies around this copy.
    //
    // The specification also requires a non-zero size and both ranges to lie
    // inside their buffers; those are checked here rather than left to the
    // validation layer, because the offsets come from a staging suballocation
    // rather than from a literal.
    //
    // Copying between two ranges of one buffer is rejected outright. The
    // specification permits it for non-overlapping regions, and nothing here
    // needs it, so the overlap rule is not implemented.
    pub fn recordCopyFrom(
        self: *const Buffer,
        source: *const Buffer,
        command_buffer: vk.CommandBuffer,
        region: CopyRegion,
    ) CopyError!void {
        try validateCopy(self.describe(), source.describe(), region);

        self.context.device.cmdCopyBuffer(
            command_buffer,
            source.handle,
            self.handle,
            &.{.{
                .src_offset = region.source_offset,
                .dst_offset = region.destination_offset,
                .size = region.size,
            }},
        );
    }
};
