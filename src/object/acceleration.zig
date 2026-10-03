const std = @import("std");
const vk = @import("vulkan");

const buffer_module = @import("buffer.zig");
const Buffer = buffer_module.Buffer;
const Context = @import("../device/context.zig").Context;
const MemoryAllocator = @import("../memory/allocator.zig").MemoryAllocator;
const commands = @import("../device/commands.zig");

const position_stride: vk.DeviceSize = @sizeOf([3]f32);
// Vulkan specification, VkAccelerationStructureGeometryInstancesDataKHR:
// data.deviceAddress must be aligned to 16 bytes.
const instance_alignment: vk.DeviceSize = 16;
const box_stride: vk.DeviceSize = @sizeOf(vk.AabbPositionsKHR);
// Vulkan specification, vkCmdBuildAccelerationStructuresKHR: an AABB
// geometry's data.deviceAddress must be aligned to 8 bytes. Its stride must be
// a multiple of 8 as well (VkAccelerationStructureGeometryAabbsDataKHR).
const box_alignment: vk.DeviceSize = 8;

pub const BuildError = error{
    BoxDataOutOfBounds,
    BufferAddressUnavailable,
    DifferentDevice,
    EmptyInstances,
    IndexDataOutOfBounds,
    InvalidBoxCount,
    InvalidIndexCount,
    InvalidInstanceFlags,
    InvalidScratchAlignment,
    InvalidVertexCount,
    MisalignedBoxData,
    MissingBuildInputUsage,
    MissingDeviceAddressUsage,
    SizeOverflow,
    BottomLevelInstanceRequired,
    VertexDataOutOfBounds,
} || std.mem.Allocator.Error ||
    buffer_module.InitError || buffer_module.UploadError ||
    vk.DeviceWrapper.CreateAccelerationStructureKHRError;

pub const IndexData = struct {
    pub const Type = enum {
        uint16,
        uint32,

        fn vulkan(self: Type) vk.IndexType {
            return switch (self) {
                .uint16 => .uint16,
                .uint32 => .uint32,
            };
        }

        fn byteSize(self: Type) vk.DeviceSize {
            return switch (self) {
                .uint16 => @sizeOf(u16),
                .uint32 => @sizeOf(u32),
            };
        }
    };

    buffer: *const Buffer,
    count: u32,
    type: Type,
};

// Positions are tightly packed triples of f32. When indices are present, every
// index must be below vertex_count; buffer contents cannot be validated here
// and must be checked before upload.
pub const TriangleGeometry = struct {
    positions: *const Buffer,
    vertex_count: u32,
    indices: ?IndexData = null,
};

// Boxes are tightly packed `VkAabbPositionsKHR`, each with its minimum at or
// below its maximum on every axis; buffer contents cannot be validated here and
// must be checked before upload. A box is only a bound: a ray query reports a
// ray entering one as a candidate, and the shader decides whether and where it
// hits. `first` counts boxes from the buffer's start, so many structures can
// take their boxes from one buffer.
pub const BoxGeometry = struct {
    boxes: *const Buffer,
    first: u32 = 0,
    count: u32,
};

pub const identity_transform: vk.TransformMatrixKHR = .{ .matrix = .{
    .{ 1, 0, 0, 0 },
    .{ 0, 1, 0, 0 },
    .{ 0, 0, 1, 0 },
} };

// One top-level instance. The bottom-level pointer is read while the build is
// recorded and is not retained.
pub const Instance = struct {
    bottom_level: *const AccelerationStructure,
    transform: vk.TransformMatrixKHR = identity_transform,
    custom_index: u24 = 0,
    mask: u8 = std.math.maxInt(u8),
    flags: vk.GeometryInstanceFlagsKHR = .{},
};

pub const AccelerationStructure = struct {
    context: *const Context,
    handle: vk.AccelerationStructureKHR,
    backing: Buffer,
    device_address: vk.DeviceAddress,
    byte_len: vk.DeviceSize,
    kind: vk.AccelerationStructureTypeKHR,

    // Vulkan specification, vkDestroyAccelerationStructureKHR and
    // vkDestroyBuffer: submitted work using the structure must be complete.
    pub fn deinit(self: *AccelerationStructure) void {
        self.context.device.destroyAccelerationStructureKHR(self.handle, null);
        self.backing.deinit();
        self.* = undefined;
    }

    pub fn deviceAddress(self: *const AccelerationStructure) vk.DeviceAddress {
        return self.device_address;
    }

    pub fn size(self: *const AccelerationStructure) vk.DeviceSize {
        return self.byte_len;
    }
};

// Scratch and top-level instance data must remain alive until the recorded build
// completes. finish releases those temporary buffers and transfers the finished
// acceleration structure to the caller.
pub const PendingBuild = struct {
    structure: AccelerationStructure,
    scratch: Buffer,
    instance_data: ?Buffer = null,

    // The command buffer carrying this build must have completed.
    pub fn finish(self: *PendingBuild) AccelerationStructure {
        self.scratch.deinit();
        if (self.instance_data) |*instance_data| instance_data.deinit();
        const structure = self.structure;
        self.* = undefined;
        return structure;
    }

    // The command buffer was not submitted, or its submission has completed.
    pub fn deinit(self: *PendingBuild) void {
        self.scratch.deinit();
        if (self.instance_data) |*instance_data| instance_data.deinit();
        self.structure.deinit();
        self.* = undefined;
    }
};

// Records one opaque triangle geometry. The position and optional index buffers
// must remain alive until the command buffer completes.
pub fn recordBottomLevel(
    context: *const Context,
    memory_allocator: *MemoryAllocator,
    command_buffer: vk.CommandBuffer,
    geometry: TriangleGeometry,
) BuildError!PendingBuild {
    try validateBuildContext(context, memory_allocator);
    if (geometry.vertex_count == 0 or
        (geometry.indices == null and geometry.vertex_count % 3 != 0))
    {
        return error.InvalidVertexCount;
    }
    const position_bytes = std.math.mul(
        vk.DeviceSize,
        geometry.vertex_count,
        position_stride,
    ) catch return error.SizeOverflow;
    if (position_bytes > geometry.positions.size) return error.VertexDataOutOfBounds;
    const position_address = try buildInputAddress(context, geometry.positions);

    const primitive_count: u32, const index_type: vk.IndexType, const index_address: vk.DeviceAddress =
        if (geometry.indices) |indices| indexed: {
            if (indices.count == 0 or indices.count % 3 != 0)
                return error.InvalidIndexCount;
            const index_bytes = std.math.mul(
                vk.DeviceSize,
                indices.count,
                indices.type.byteSize(),
            ) catch return error.SizeOverflow;
            if (index_bytes > indices.buffer.size) return error.IndexDataOutOfBounds;
            break :indexed .{
                indices.count / 3,
                indices.type.vulkan(),
                try buildInputAddress(context, indices.buffer),
            };
        } else .{ geometry.vertex_count / 3, .none_khr, 0 };

    const vk_geometry = vk.AccelerationStructureGeometryKHR{
        .geometry_type = .triangles_khr,
        .geometry = .{ .triangles = .{
            .vertex_format = .r32g32b32_sfloat,
            .vertex_data = .{ .device_address = position_address },
            .vertex_stride = position_stride,
            .max_vertex = geometry.vertex_count - 1,
            .index_type = index_type,
            .index_data = .{ .device_address = index_address },
            .transform_data = .{ .device_address = 0 },
        } },
        .flags = .{ .opaque_bit_khr = true },
    };
    return recordBuild(
        context,
        memory_allocator,
        command_buffer,
        .bottom_level_khr,
        vk_geometry,
        primitive_count,
    );
}

// Records one opaque box geometry. The box buffer must remain alive until the
// command buffer completes.
pub fn recordBottomLevelBoxes(
    context: *const Context,
    memory_allocator: *MemoryAllocator,
    command_buffer: vk.CommandBuffer,
    geometry: BoxGeometry,
) BuildError!PendingBuild {
    comptime std.debug.assert(box_stride % box_alignment == 0);
    try validateBuildContext(context, memory_allocator);
    if (geometry.count == 0) return error.InvalidBoxCount;
    const end = std.math.add(u64, geometry.first, geometry.count) catch return error.SizeOverflow;
    const end_bytes = std.math.mul(vk.DeviceSize, end, box_stride) catch return error.SizeOverflow;
    if (end_bytes > geometry.boxes.size) return error.BoxDataOutOfBounds;
    // The start stays aligned when the buffer's is: the stride is a multiple
    // of the alignment.
    const box_address = std.math.add(
        vk.DeviceAddress,
        try buildInputAddress(context, geometry.boxes),
        @as(vk.DeviceSize, geometry.first) * box_stride,
    ) catch return error.SizeOverflow;
    if (box_address % box_alignment != 0) return error.MisalignedBoxData;

    const vk_geometry = vk.AccelerationStructureGeometryKHR{
        .geometry_type = .aabbs_khr,
        .geometry = .{ .aabbs = .{
            .data = .{ .device_address = box_address },
            .stride = box_stride,
        } },
        .flags = .{ .opaque_bit_khr = true },
    };
    return recordBuild(
        context,
        memory_allocator,
        command_buffer,
        .bottom_level_khr,
        vk_geometry,
        geometry.count,
    );
}

// Records one top-level structure from a contiguous instance array. The array is
// copied into an owned upload buffer and may be released when this function
// returns.
pub fn recordTopLevel(
    context: *const Context,
    memory_allocator: *MemoryAllocator,
    command_buffer: vk.CommandBuffer,
    instances: []const Instance,
) BuildError!PendingBuild {
    try validateBuildContext(context, memory_allocator);
    if (instances.len == 0) return error.EmptyInstances;
    if (instances.len > std.math.maxInt(u32)) return error.SizeOverflow;

    for (instances) |instance| {
        if (instance.bottom_level.context.device.handle != context.device.handle)
            return error.DifferentDevice;
        if (instance.bottom_level.kind != .bottom_level_khr)
            return error.BottomLevelInstanceRequired;
        if (instance.flags.toInt() > std.math.maxInt(u8))
            return error.InvalidInstanceFlags;
    }

    const raw_instances = try memory_allocator.host_allocator.alloc(
        vk.AccelerationStructureInstanceKHR,
        instances.len,
    );
    defer memory_allocator.host_allocator.free(raw_instances);
    for (instances, raw_instances) |instance, *raw| {
        raw.* = .{
            .transform = instance.transform,
            .instance_custom_index_and_mask = .{
                .instance_custom_index = instance.custom_index,
                .mask = instance.mask,
            },
            .instance_shader_binding_table_record_offset_and_flags = .{
                .instance_shader_binding_table_record_offset = 0,
                .flags = @intCast(instance.flags.toInt()),
            },
            .acceleration_structure_reference = instance.bottom_level.device_address,
        };
    }

    const instance_bytes = std.math.mul(
        vk.DeviceSize,
        @as(vk.DeviceSize, @intCast(instances.len)),
        @as(vk.DeviceSize, @sizeOf(vk.AccelerationStructureInstanceKHR)),
    ) catch return error.SizeOverflow;
    const instance_buffer_size = std.math.add(
        vk.DeviceSize,
        instance_bytes,
        instance_alignment - 1,
    ) catch return error.SizeOverflow;
    var instance_data = try Buffer.init(
        context,
        memory_allocator,
        instance_buffer_size,
        .{
            .shader_device_address_bit = true,
            .acceleration_structure_build_input_read_only_bit_khr = true,
        },
        .upload,
    );
    errdefer instance_data.deinit();
    const instance_address = try alignedAddress(
        try bufferAddress(context, &instance_data),
        instance_alignment,
    );
    try instance_data.uploadAt(
        instance_address.offset,
        std.mem.sliceAsBytes(raw_instances),
    );

    const geometry = vk.AccelerationStructureGeometryKHR{
        .geometry_type = .instances_khr,
        .geometry = .{ .instances = .{
            .array_of_pointers = .false,
            .data = .{ .device_address = instance_address.address },
        } },
    };
    var pending = try recordBuild(
        context,
        memory_allocator,
        command_buffer,
        .top_level_khr,
        geometry,
        @intCast(instances.len),
    );
    pending.instance_data = instance_data;
    return pending;
}

fn recordBuild(
    context: *const Context,
    memory_allocator: *MemoryAllocator,
    command_buffer: vk.CommandBuffer,
    kind: vk.AccelerationStructureTypeKHR,
    geometry: vk.AccelerationStructureGeometryKHR,
    primitive_count: u32,
) BuildError!PendingBuild {
    if (!context.ray_query_enabled) return error.RayQueryDisabled;
    if (memory_allocator.context.device.handle != context.device.handle)
        return error.AllocatorDeviceMismatch;

    const properties = queryProperties(context);
    switch (kind) {
        .bottom_level_khr => if (primitive_count > properties.max_primitive_count)
            return error.SizeLimitExceeded,
        .top_level_khr => if (primitive_count > properties.max_instance_count)
            return error.SizeLimitExceeded,
        else => unreachable,
    }

    var build_info = vk.AccelerationStructureBuildGeometryInfoKHR{
        .type = kind,
        .flags = .{ .prefer_fast_trace_bit_khr = true },
        .mode = .build_khr,
        .geometry_count = 1,
        .p_geometries = @ptrCast(&geometry),
        .scratch_data = .{ .device_address = 0 },
    };
    var sizes: vk.AccelerationStructureBuildSizesInfoKHR = undefined;
    sizes.s_type = .acceleration_structure_build_sizes_info_khr;
    sizes.p_next = null;
    context.device.getAccelerationStructureBuildSizesKHR(
        .device_khr,
        &build_info,
        @ptrCast(&primitive_count),
        &sizes,
    );

    var backing = try Buffer.init(
        context,
        memory_allocator,
        sizes.acceleration_structure_size,
        .{
            .shader_device_address_bit = true,
            .acceleration_structure_storage_bit_khr = true,
        },
        .device,
    );
    errdefer backing.deinit();

    const handle = try context.device.createAccelerationStructureKHR(&.{
        .buffer = backing.handle,
        .offset = 0,
        .size = sizes.acceleration_structure_size,
        .type = kind,
    }, null);
    errdefer context.device.destroyAccelerationStructureKHR(handle, null);

    const device_address = context.device.getAccelerationStructureDeviceAddressKHR(&.{
        .acceleration_structure = handle,
    });
    if (device_address == 0) return error.BufferAddressUnavailable;

    const scratch_alignment: vk.DeviceSize = properties.min_acceleration_structure_scratch_offset_alignment;
    if (scratch_alignment == 0) return error.InvalidScratchAlignment;
    const scratch_size = std.math.add(
        vk.DeviceSize,
        sizes.build_scratch_size,
        scratch_alignment - 1,
    ) catch return error.SizeOverflow;
    var scratch = try Buffer.init(
        context,
        memory_allocator,
        scratch_size,
        .{
            .storage_buffer_bit = true,
            .shader_device_address_bit = true,
        },
        .device,
    );
    errdefer scratch.deinit();

    const scratch_address = try alignedAddress(
        try bufferAddress(context, &scratch),
        scratch_alignment,
    );

    build_info.dst_acceleration_structure = handle;
    build_info.scratch_data = .{ .device_address = scratch_address.address };
    const range = vk.AccelerationStructureBuildRangeInfoKHR{
        .primitive_count = primitive_count,
        .primitive_offset = 0,
        .first_vertex = 0,
        .transform_offset = 0,
    };
    const range_pointers = [_][*]const vk.AccelerationStructureBuildRangeInfoKHR{
        @ptrCast(&range),
    };
    context.device.cmdBuildAccelerationStructuresKHR(
        command_buffer,
        &.{build_info},
        &range_pointers,
    );
    // Vulkan specification, "Building Acceleration Structures": a top level
    // cannot be built in the same call as the bottom levels its instances name,
    // since builds in one call are not ordered. Built in a later call it is
    // ordered only by a dependency, so the build stage waits here beside the
    // shaders that trace through the structure.
    commands.recordMemoryBarrier(context, command_buffer, .{
        .src_stage = .{ .acceleration_structure_build_bit_khr = true },
        .src_access = .{ .acceleration_structure_write_bit_khr = true },
        .dst_stage = .{
            .acceleration_structure_build_bit_khr = true,
            .compute_shader_bit = true,
            .fragment_shader_bit = true,
        },
        .dst_access = .{ .acceleration_structure_read_bit_khr = true },
    });

    return .{
        .structure = .{
            .context = context,
            .handle = handle,
            .backing = backing,
            .device_address = device_address,
            .byte_len = sizes.acceleration_structure_size,
            .kind = kind,
        },
        .scratch = scratch,
    };
}

const AlignedAddress = struct {
    address: vk.DeviceAddress,
    offset: vk.DeviceSize,
};

fn alignedAddress(base: vk.DeviceAddress, alignment: vk.DeviceSize) BuildError!AlignedAddress {
    if (alignment == 0) return error.InvalidScratchAlignment;
    const remainder = base % alignment;
    const offset = if (remainder == 0) 0 else alignment - remainder;
    return .{
        .address = std.math.add(vk.DeviceAddress, base, offset) catch
            return error.SizeOverflow,
        .offset = offset,
    };
}

fn validateBuildContext(
    context: *const Context,
    memory_allocator: *const MemoryAllocator,
) BuildError!void {
    if (!context.ray_query_enabled) return error.RayQueryDisabled;
    if (memory_allocator.context.device.handle != context.device.handle)
        return error.AllocatorDeviceMismatch;
}

fn buildInputAddress(context: *const Context, buffer: *const Buffer) BuildError!vk.DeviceAddress {
    if (buffer.context.device.handle != context.device.handle)
        return error.DifferentDevice;
    if (!buffer.usage.acceleration_structure_build_input_read_only_bit_khr)
        return error.MissingBuildInputUsage;
    return bufferAddress(context, buffer);
}

fn bufferAddress(context: *const Context, buffer: *const Buffer) BuildError!vk.DeviceAddress {
    if (buffer.context.device.handle != context.device.handle)
        return error.DifferentDevice;
    if (!buffer.usage.shader_device_address_bit)
        return error.MissingDeviceAddressUsage;
    const address = context.device.getBufferDeviceAddress(&.{ .buffer = buffer.handle });
    return if (address == 0) error.BufferAddressUnavailable else address;
}

fn queryProperties(context: *const Context) vk.PhysicalDeviceAccelerationStructurePropertiesKHR {
    var acceleration: vk.PhysicalDeviceAccelerationStructurePropertiesKHR = undefined;
    acceleration.s_type = .physical_device_acceleration_structure_properties_khr;
    acceleration.p_next = null;
    var properties = vk.PhysicalDeviceProperties2{
        .p_next = @ptrCast(&acceleration),
        .properties = undefined,
    };
    context.instance.getPhysicalDeviceProperties2(context.physical_device, &properties);
    return acceleration;
}
