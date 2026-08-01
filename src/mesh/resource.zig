const std = @import("std");
const vk = @import("vulkan");

const bounds = @import("bounds.zig");
const buffer_module = @import("../buffer.zig");
const Context = @import("../context.zig").Context;
const memory = @import("../memory/allocator.zig");
const staging = @import("../staging/arena.zig");
const vertex_module = @import("vertex.zig");

const Buffer = buffer_module.Buffer;
const StagingArena = staging.StagingArena;
const Vertex3D = vertex_module.Vertex3D;
const VertexStreams = vertex_module.VertexStreams;

// Every stream is laid out at this alignment inside one staging reservation.
// vkCmdCopyBuffer constrains no offset, so this exists to keep each stream's
// elements naturally aligned for the host writes that fill them.
const stream_alignment: vk.DeviceSize = 4;

// Every buffer a mesh can own: the mandatory vertex stream, one per optional
// stream, morph deltas and indices. Derived from the stream set so adding a
// stream moves it.
const max_streams = 1 + @typeInfo(VertexStreams).@"struct".fields.len + 2;

pub const InitError = error{
    EmptyMesh,
    IndexOutOfRange,
    MorphDataMismatch,
    TooManyVertices,
} || staging.ReserveError || buffer_module.InitError || buffer_module.CopyError;

// Morph target deltas as an asset supplies them: one array of positions and
// optionally one of normals, each three floats per (vertex, target) pair with
// the target index varying fastest. They are interleaved into the GPU layout
// during upload.
pub const MorphUpload = struct {
    positions: []const f32,
    normals: []const f32 = &.{},
    target_count: u32,
};

// One morph element as the vertex shader reads it: a position delta and a normal
// delta. Interleaving them means the shader indexes with the target count alone.
const MorphElement = extern struct {
    position: [3]f32,
    normal: [3]f32,

    comptime {
        std.debug.assert(@sizeOf(MorphElement) == 24);
    }
};

pub fn Upload(comptime IndexType: type) type {
    comptime assertIndexType(IndexType);
    return struct {
        vertices: []const Vertex3D,
        indices: ?[]const IndexType = null,
        streams: VertexStreams = .{},
        morph: ?MorphUpload = null,
    };
}

pub const Mesh = struct {
    vertex_buffer: Buffer,
    // Present exactly when the corresponding stream flag is set.
    skin_buffer: ?Buffer,
    colour_buffer: ?Buffer,
    uv1_buffer: ?Buffer,
    // Present when the mesh has morph targets. Element v * target_count + t
    // holds target t's deltas for vertex v.
    morph_buffer: ?Buffer,
    index_buffer: ?Buffer,
    vertex_count: u32,
    index_count: u32,
    index_type: vk.IndexType,
    streams: VertexStreams,
    morph_target_count: u32,
    bounds: bounds.Bounds,

    // Packs the interchange vertices straight into the staging arena and records
    // one copy per stream into the caller's command buffer. Nothing is submitted
    // here, and the arena's contents must stay untouched until that command
    // buffer completes.
    //
    // The whole staging requirement is reserved before any Vulkan object exists,
    // so an arena too full to hold this mesh fails without leaving a partially
    // built one behind. The caller answers OutOfSpace by submitting what it has,
    // waiting, resetting the arena and calling again; the reservation this call
    // did make is not returned, because an arena reclaims everything at once.
    pub fn init(
        comptime IndexType: type,
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        arena: *StagingArena,
        command_buffer: vk.CommandBuffer,
        upload: Upload(IndexType),
    ) InitError!Mesh {
        comptime assertIndexType(IndexType);
        const vertices = upload.vertices;
        if (vertices.len == 0) return error.EmptyMesh;
        if (vertices.len > std.math.maxInt(u32)) return error.TooManyVertices;
        const vertex_count: u32 = @intCast(vertices.len);

        const morph_target_count = try validateMorph(upload.morph, vertices.len);
        if (upload.indices) |indices| {
            // Indices come from an asset and address the vertex fetch directly.
            // An out-of-range one reads outside the bound buffer, which is a
            // hang or a device loss rather than a wrong picture.
            for (indices) |index| {
                if (@as(u64, index) >= vertices.len) return error.IndexOutOfRange;
            }
        }

        var plan: Plan = .{};
        plan.add(vertex_count * @sizeOf(vertex_module.GpuVertex));
        if (upload.streams.skinned)
            plan.add(vertex_count * @sizeOf(vertex_module.GpuSkinVertex));
        if (upload.streams.colour)
            plan.add(vertex_count * @sizeOf(vertex_module.GpuColourVertex));
        if (upload.streams.uv1)
            plan.add(vertex_count * @sizeOf(vertex_module.GpuUv1Vertex));
        if (morph_target_count > 0)
            plan.add(@as(vk.DeviceSize, vertex_count) * morph_target_count * @sizeOf(MorphElement));
        if (upload.indices) |indices|
            plan.add(indices.len * @sizeOf(IndexType));

        // The single point of failure for staging capacity. Past this line every
        // stream has its bytes, and only device allocation can still fail.
        const reservation = try arena.reserve(plan.total, stream_alignment);
        var writer: Writer = .{ .arena = arena, .reservation = reservation };

        // Each buffer is owned by a local with its own rollback from the moment
        // it exists. Nothing here reads a field that has not been assigned.
        const packed_vertices = writer.take(vertex_module.GpuVertex, vertex_count);
        for (vertices, packed_vertices) |*source, *target|
            target.* = vertex_module.packVertex(source);
        var vertex_buffer = try writer.commit(
            context,
            memory_allocator,
            command_buffer,
            packed_vertices,
            .{ .vertex_buffer_bit = true },
        );
        errdefer vertex_buffer.deinit();

        var skin_buffer: ?Buffer = null;
        errdefer if (skin_buffer) |*owned| owned.deinit();
        if (upload.streams.skinned) {
            const packed_skin = writer.take(vertex_module.GpuSkinVertex, vertex_count);
            for (vertices, packed_skin) |*source, *target|
                target.* = vertex_module.packSkinVertex(source);
            skin_buffer = try writer.commit(
                context,
                memory_allocator,
                command_buffer,
                packed_skin,
                .{ .vertex_buffer_bit = true },
            );
        }

        var colour_buffer: ?Buffer = null;
        errdefer if (colour_buffer) |*owned| owned.deinit();
        if (upload.streams.colour) {
            const packed_colour = writer.take(vertex_module.GpuColourVertex, vertex_count);
            for (vertices, packed_colour) |*source, *target|
                target.* = vertex_module.packColourVertex(source);
            colour_buffer = try writer.commit(
                context,
                memory_allocator,
                command_buffer,
                packed_colour,
                .{ .vertex_buffer_bit = true },
            );
        }

        var uv1_buffer: ?Buffer = null;
        errdefer if (uv1_buffer) |*owned| owned.deinit();
        if (upload.streams.uv1) {
            const packed_uv1 = writer.take(vertex_module.GpuUv1Vertex, vertex_count);
            for (vertices, packed_uv1) |*source, *target|
                target.* = vertex_module.packUv1Vertex(source);
            uv1_buffer = try writer.commit(
                context,
                memory_allocator,
                command_buffer,
                packed_uv1,
                .{ .vertex_buffer_bit = true },
            );
        }

        var morph_buffer: ?Buffer = null;
        errdefer if (morph_buffer) |*owned| owned.deinit();
        if (morph_target_count > 0) {
            const morph = upload.morph.?;
            const elements = writer.take(MorphElement, vertex_count * morph_target_count);
            const has_normals = morph.normals.len > 0;
            for (elements, 0..) |*element, index| {
                element.* = .{
                    .position = morph.positions[index * 3 ..][0..3].*,
                    .normal = if (has_normals)
                        morph.normals[index * 3 ..][0..3].*
                    else
                        .{ 0, 0, 0 },
                };
            }
            morph_buffer = try writer.commit(
                context,
                memory_allocator,
                command_buffer,
                elements,
                .{ .storage_buffer_bit = true },
            );
        }

        var index_buffer: ?Buffer = null;
        var index_count: u32 = 0;
        errdefer if (index_buffer) |*owned| owned.deinit();
        if (upload.indices) |indices| {
            const staged = writer.take(IndexType, @intCast(indices.len));
            @memcpy(staged, indices);
            index_buffer = try writer.commit(
                context,
                memory_allocator,
                command_buffer,
                staged,
                .{ .index_buffer_bit = true },
            );
            index_count = @intCast(indices.len);
        }

        return .{
            .vertex_buffer = vertex_buffer,
            .skin_buffer = skin_buffer,
            .colour_buffer = colour_buffer,
            .uv1_buffer = uv1_buffer,
            .morph_buffer = morph_buffer,
            .index_buffer = index_buffer,
            .vertex_count = vertex_count,
            .index_count = index_count,
            .index_type = comptime if (IndexType == u16) .uint16 else .uint32,
            .streams = upload.streams,
            .morph_target_count = morph_target_count,
            .bounds = bounds.Bounds.compute(vertices),
        };
    }

    // Vulkan specification, vkDestroyBuffer: submitted work reading this mesh
    // must have completed.
    pub fn deinit(self: *Mesh) void {
        if (self.index_buffer) |*owned| owned.deinit();
        if (self.morph_buffer) |*owned| owned.deinit();
        if (self.uv1_buffer) |*owned| owned.deinit();
        if (self.colour_buffer) |*owned| owned.deinit();
        if (self.skin_buffer) |*owned| owned.deinit();
        self.vertex_buffer.deinit();
        self.* = undefined;
    }

    // Binds each present stream at its own binding. The optional ones sit at
    // fixed, non-contiguous bindings and appear independently, so they cannot be
    // bound as one range.
    pub fn bind(self: *const Mesh, context: *const Context, command_buffer: vk.CommandBuffer) void {
        const start: [1]vk.DeviceSize = .{0};
        context.device.cmdBindVertexBuffers(command_buffer, 0, &.{self.vertex_buffer.handle}, &start);
        if (self.skin_buffer) |owned|
            context.device.cmdBindVertexBuffers(command_buffer, 1, &.{owned.handle}, &start);
        if (self.colour_buffer) |owned|
            context.device.cmdBindVertexBuffers(command_buffer, 2, &.{owned.handle}, &start);
        if (self.uv1_buffer) |owned|
            context.device.cmdBindVertexBuffers(command_buffer, 3, &.{owned.handle}, &start);
        if (self.index_buffer) |owned|
            context.device.cmdBindIndexBuffer(command_buffer, owned.handle, 0, self.index_type);
    }

    pub fn draw(self: *const Mesh, context: *const Context, command_buffer: vk.CommandBuffer) void {
        if (self.index_count > 0) {
            context.device.cmdDrawIndexed(command_buffer, self.index_count, 1, 0, 0, 0);
        } else {
            context.device.cmdDraw(command_buffer, self.vertex_count, 1, 0, 0);
        }
    }
};

// Sums what the streams need, at the same alignment the writer places them at,
// so the reservation is exactly large enough and the placement below cannot run
// past it.
const Plan = struct {
    total: vk.DeviceSize = 0,
    count: usize = 0,

    fn add(self: *Plan, bytes: vk.DeviceSize) void {
        self.total += std.mem.alignForward(vk.DeviceSize, bytes, stream_alignment);
        self.count += 1;
        std.debug.assert(self.count <= max_streams);
    }
};

// Hands out typed slices of the reservation in the order Plan sized them, and
// records the copy for each. Both walk the same sequence, so the offsets agree
// by construction rather than by two matching calculations.
const Writer = struct {
    arena: *const StagingArena,
    reservation: staging.Reservation,
    used: vk.DeviceSize = 0,
    stream_offset: vk.DeviceSize = 0,

    fn take(self: *Writer, comptime T: type, count: u32) []T {
        self.stream_offset = self.used;
        const bytes = @as(vk.DeviceSize, count) * @sizeOf(T);
        const slice = self.reservation.bytes[@intCast(self.used)..][0..@intCast(bytes)];
        self.used += std.mem.alignForward(vk.DeviceSize, bytes, stream_alignment);
        return @alignCast(std.mem.bytesAsSlice(T, slice));
    }

    fn commit(
        self: *Writer,
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        command_buffer: vk.CommandBuffer,
        contents: anytype,
        usage: vk.BufferUsageFlags,
    ) !Buffer {
        const bytes: vk.DeviceSize = @intCast(std.mem.sliceAsBytes(contents).len);
        var device_buffer = try Buffer.init(
            context,
            memory_allocator,
            bytes,
            usage.merge(.{ .transfer_dst_bit = true }),
            .device,
        );
        errdefer device_buffer.deinit();

        try device_buffer.recordCopyFrom(self.arena.source(), command_buffer, .{
            .source_offset = self.reservation.offset + self.stream_offset,
            .size = bytes,
        });
        return device_buffer;
    }
};

fn validateMorph(morph: ?MorphUpload, vertex_count: usize) InitError!u32 {
    const upload = morph orelse return 0;
    if (upload.target_count == 0) return 0;

    const pairs = std.math.mul(usize, vertex_count, upload.target_count) catch
        return error.MorphDataMismatch;
    const components = std.math.mul(usize, pairs, 3) catch return error.MorphDataMismatch;
    if (upload.positions.len != components) return error.MorphDataMismatch;
    if (upload.normals.len != 0 and upload.normals.len != components)
        return error.MorphDataMismatch;
    return upload.target_count;
}

fn assertIndexType(comptime IndexType: type) void {
    if (IndexType != u16 and IndexType != u32)
        @compileError("a mesh index is u16 or u32");
}
