const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;

// Handles are opaque to these checks: they are only compared, never
// dereferenced, so synthetic ones are enough to exercise every decision the
// policy makes.
const device: vk.Device = @fromBackingInt(@intCast(1));
const other_device: vk.Device = @fromBackingInt(@intCast(2));
const source_handle: vk.Buffer = @fromBackingInt(@intCast(10));
const destination_handle: vk.Buffer = @fromBackingInt(@intCast(11));

const max_buffer_size: vk.DeviceSize = 4096;

// A request that passes every check, which each test below alters in one place.
const valid_request: gpu.BufferCreateRequest = .{
    .device = device,
    .allocator_device = device,
    .size = 256,
    .max_buffer_size = max_buffer_size,
    .usage = .{ .transfer_dst_bit = true },
    .ray_query_enabled = false,
};

fn source(size: vk.DeviceSize, usage: vk.BufferUsageFlags) gpu.BufferDescriptor {
    return .{ .device = device, .handle = source_handle, .size = size, .usage = usage };
}

fn destination(size: vk.DeviceSize, usage: vk.BufferUsageFlags) gpu.BufferDescriptor {
    return .{ .device = device, .handle = destination_handle, .size = size, .usage = usage };
}

test "creation accepts a supported buffer" {
    try gpu.validateBufferCreate(valid_request);

    var largest = valid_request;
    largest.size = max_buffer_size;
    try gpu.validateBufferCreate(largest);

    var combined = valid_request;
    combined.usage = .{ .vertex_buffer_bit = true, .transfer_dst_bit = true };
    try gpu.validateBufferCreate(combined);
}

test "creation refuses sizes outside the device's range" {
    var empty = valid_request;
    empty.size = 0;
    try testing.expectError(error.InvalidSize, gpu.validateBufferCreate(empty));

    var oversized = valid_request;
    oversized.size = max_buffer_size + 1;
    try testing.expectError(error.SizeLimitExceeded, gpu.validateBufferCreate(oversized));
}

test "creation refuses usage this module does not support" {
    var none = valid_request;
    none.usage = .{};
    try testing.expectError(error.EmptyUsage, gpu.validateBufferCreate(none));

    var unsupported = valid_request;
    unsupported.usage = .{ .conditional_rendering_bit_ext = true };
    try testing.expectError(error.UnsupportedUsage, gpu.validateBufferCreate(unsupported));
}

test "ray query buffer usages require negotiated support" {
    const usages = [_]vk.BufferUsageFlags{
        .{ .shader_device_address_bit = true },
        .{ .acceleration_structure_build_input_read_only_bit_khr = true },
        .{ .acceleration_structure_storage_bit_khr = true },
    };
    for (usages) |usage| {
        var request = valid_request;
        request.usage = usage;
        try testing.expectError(error.RayQueryDisabled, gpu.validateBufferCreate(request));

        request.ray_query_enabled = true;
        try gpu.validateBufferCreate(request);
    }
}

test "creation refuses an allocator belonging to another device" {
    var mismatched = valid_request;
    mismatched.allocator_device = other_device;
    try testing.expectError(error.AllocatorDeviceMismatch, gpu.validateBufferCreate(mismatched));
}

test "a copy within both buffers is accepted" {
    const src = source(256, .{ .transfer_src_bit = true });
    const dst = destination(256, .{ .transfer_dst_bit = true });

    try gpu.validateBufferCopy(dst, src, .{ .size = 256 });
    try gpu.validateBufferCopy(dst, src, .{ .size = 1, .source_offset = 255 });
    try gpu.validateBufferCopy(dst, src, .{ .size = 1, .destination_offset = 255 });
    try gpu.validateBufferCopy(dst, src, .{
        .size = 128,
        .source_offset = 128,
        .destination_offset = 0,
    });
}

test "a copy needs the transfer usages on the right ends" {
    const src = source(256, .{ .transfer_src_bit = true });
    const dst = destination(256, .{ .transfer_dst_bit = true });

    try testing.expectError(error.MissingTransferSourceUsage, gpu.validateBufferCopy(
        dst,
        source(256, .{ .transfer_dst_bit = true }),
        .{ .size = 16 },
    ));
    try testing.expectError(error.MissingTransferDestinationUsage, gpu.validateBufferCopy(
        destination(256, .{ .transfer_src_bit = true }),
        src,
        .{ .size = 16 },
    ));
}

test "a copy is refused across devices and within one buffer" {
    const src = source(256, .{ .transfer_src_bit = true });
    const dst = destination(256, .{ .transfer_dst_bit = true });

    var elsewhere = src;
    elsewhere.device = other_device;
    try testing.expectError(
        error.DifferentDevice,
        gpu.validateBufferCopy(dst, elsewhere, .{ .size = 16 }),
    );

    var same = dst;
    same.handle = source_handle;
    same.usage = .{ .transfer_src_bit = true, .transfer_dst_bit = true };
    try testing.expectError(
        error.SameBuffer,
        gpu.validateBufferCopy(same, src, .{ .size = 16 }),
    );
}

// The range tests are subtractions rather than additions so that an offset near
// the end of a buffer cannot overflow into looking valid. These are the cases
// that would pass if they were written the other way.
test "a copy range may not leave either buffer" {
    const src = source(256, .{ .transfer_src_bit = true });
    const dst = destination(256, .{ .transfer_dst_bit = true });

    try testing.expectError(
        error.InvalidSize,
        gpu.validateBufferCopy(dst, src, .{ .size = 0 }),
    );
    try testing.expectError(
        error.SourceRangeOutOfBounds,
        gpu.validateBufferCopy(dst, src, .{ .size = 257 }),
    );
    try testing.expectError(
        error.SourceRangeOutOfBounds,
        gpu.validateBufferCopy(dst, src, .{ .size = 1, .source_offset = 256 }),
    );
    try testing.expectError(
        error.SourceRangeOutOfBounds,
        gpu.validateBufferCopy(dst, src, .{
            .size = 16,
            .source_offset = std.math.maxInt(vk.DeviceSize) - 8,
        }),
    );
    try testing.expectError(
        error.DestinationRangeOutOfBounds,
        gpu.validateBufferCopy(dst, source(1024, .{ .transfer_src_bit = true }), .{ .size = 512 }),
    );
    try testing.expectError(
        error.DestinationRangeOutOfBounds,
        gpu.validateBufferCopy(dst, src, .{
            .size = 16,
            .destination_offset = std.math.maxInt(vk.DeviceSize) - 8,
        }),
    );
}
