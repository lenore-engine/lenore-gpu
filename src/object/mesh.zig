const std = @import("std");
const vk = @import("vulkan");

const res = @import("lenore-resources");
const buffer_module = @import("buffer.zig");
const Context = @import("../device/context.zig").Context;
const memory = @import("../memory/allocator.zig");
const staging = @import("../staging/pool.zig");
const transfer_module = @import("../staging/transfer.zig");
const vertex_module = @import("vertex.zig");

const Buffer = buffer_module.Buffer;
const Transfer = transfer_module.Transfer;
const Vertex3D = res.Vertex3D;
const VertexStreams = res.VertexStreams;

// Every chunk of a stream starts at this alignment inside a staging block.
// vkCmdCopyBuffer constrains no offset, so this exists for the host writes: the
// packing loops below write typed elements straight into the mapped bytes. It
// is the pool's own guarantee rather than a second four, because an offset
// aligned here into a block whose base is not gives an unaligned element.
const stream_alignment: vk.DeviceSize = staging.host_element_alignment;

pub const InitError = error{
    EmptyMesh,
    IndexOutOfRange,
    MorphDataMismatch,
    TooManyVertices,
} || StreamError || buffer_module.InitError;

const StreamError = transfer_module.ReserveError || buffer_module.CopyError;

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

// What the base vertex buffer is created with.
//
// A morphed mesh's vertices are read by the prepass as a storage buffer, so it
// carries that usage as well. Conditional rather than always set: Vulkan lets an
// implementation restrict which memory types a buffer may be placed in by its
// usage, and asking for one nothing reads spends that for nothing.
//
// Split from the creation for the reason the other validators here are: a
// device stands between this decision and a test, and choosing wrong is a
// descriptor the layer refuses at registration rather than anything visible.
pub fn vertexUsage(morph_target_count: u32) vk.BufferUsageFlags {
    return .{
        .vertex_buffer_bit = true,
        .storage_buffer_bit = morph_target_count > 0,
    };
}

// Where a draw fetches binding zero from. Ordinarily the mesh's own buffer, and
// for a morphed mesh the prepass's output for the frame being recorded.
//
// It lives here rather than beside the prepass because this is the type `bind`
// takes, and a mesh knows nothing about what produced an alternative source.
pub const VertexSource = struct {
    handle: vk.Buffer,
    offset: vk.DeviceSize,
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
    bounds: res.Bounds,

    // Packs the interchange vertices straight into staging memory and records
    // the copies that fill each stream's device buffer.
    //
    // A stream is written in whatever chunks the staging pool hands out, so no
    // mesh is too large for it and no size is reserved up front. Reclaiming that
    // pool means submitting, which the transfer does on its own, so this call
    // can submit and wait partway through. What it never does is submit the last
    // of the work: that is the caller's, and until it happens the device buffers
    // returned here are only partly filled.
    //
    // Everything an asset can get wrong is checked before the first device
    // object exists, so a rejected mesh leaves nothing behind.
    pub fn init(
        comptime IndexType: type,
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        transfer: *Transfer,
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

        // One rollback for every buffer, because the decision it takes is one
        // policy rather than six. See Built.
        var built: Built = .{};
        errdefer built.rollback(transfer);

        built.vertex = try deviceBuffer(
            context,
            memory_allocator,
            vertex_module.GpuVertex,
            vertex_count,
            vertexUsage(morph_target_count),
        );
        var vertex_chunks = Chunks(vertex_module.GpuVertex).init(
            transfer,
            &built.vertex.?,
            vertex_count,
        );
        while (try vertex_chunks.next()) |chunk| {
            for (vertices[chunk.first..][0..chunk.items.len], chunk.items) |*source, *target|
                target.* = vertex_module.packVertex(source);
        }

        if (upload.streams.skinned) {
            built.skin = try deviceBuffer(
                context,
                memory_allocator,
                vertex_module.GpuSkinVertex,
                vertex_count,
                .{ .vertex_buffer_bit = true },
            );
            var chunks = Chunks(vertex_module.GpuSkinVertex).init(
                transfer,
                &built.skin.?,
                vertex_count,
            );
            while (try chunks.next()) |chunk| {
                for (vertices[chunk.first..][0..chunk.items.len], chunk.items) |*source, *target|
                    target.* = vertex_module.packSkinVertex(source);
            }
        }

        if (upload.streams.colour) {
            built.colour = try deviceBuffer(
                context,
                memory_allocator,
                vertex_module.GpuColourVertex,
                vertex_count,
                .{ .vertex_buffer_bit = true },
            );
            var chunks = Chunks(vertex_module.GpuColourVertex).init(
                transfer,
                &built.colour.?,
                vertex_count,
            );
            while (try chunks.next()) |chunk| {
                for (vertices[chunk.first..][0..chunk.items.len], chunk.items) |*source, *target|
                    target.* = vertex_module.packColourVertex(source);
            }
        }

        if (upload.streams.uv1) {
            built.uv1 = try deviceBuffer(
                context,
                memory_allocator,
                vertex_module.GpuUv1Vertex,
                vertex_count,
                .{ .vertex_buffer_bit = true },
            );
            var chunks = Chunks(vertex_module.GpuUv1Vertex).init(
                transfer,
                &built.uv1.?,
                vertex_count,
            );
            while (try chunks.next()) |chunk| {
                for (vertices[chunk.first..][0..chunk.items.len], chunk.items) |*source, *target|
                    target.* = vertex_module.packUv1Vertex(source);
            }
        }

        if (morph_target_count > 0) {
            const morph = upload.morph.?;
            // validateMorph proved this product and its triple fit a usize, and
            // that the source arrays are exactly that long.
            const elements = vertices.len * @as(usize, morph_target_count);
            built.morph = try deviceBuffer(
                context,
                memory_allocator,
                MorphElement,
                elements,
                .{ .storage_buffer_bit = true },
            );
            const has_normals = morph.normals.len > 0;
            var chunks = Chunks(MorphElement).init(transfer, &built.morph.?, elements);
            while (try chunks.next()) |chunk| {
                for (chunk.items, chunk.first..) |*element, index| {
                    element.* = .{
                        .position = morph.positions[index * 3 ..][0..3].*,
                        .normal = if (has_normals)
                            morph.normals[index * 3 ..][0..3].*
                        else
                            .{ 0, 0, 0 },
                    };
                }
            }
        }

        var index_count: u32 = 0;
        if (upload.indices) |indices| {
            built.index = try deviceBuffer(
                context,
                memory_allocator,
                IndexType,
                indices.len,
                .{ .index_buffer_bit = true },
            );
            var chunks = Chunks(IndexType).init(transfer, &built.index.?, indices.len);
            while (try chunks.next()) |chunk| {
                @memcpy(chunk.items, indices[chunk.first..][0..chunk.items.len]);
            }
            index_count = @intCast(indices.len);
        }

        return .{
            .vertex_buffer = built.vertex.?,
            .skin_buffer = built.skin,
            .colour_buffer = built.colour,
            .uv1_buffer = built.uv1,
            .morph_buffer = built.morph,
            .index_buffer = built.index,
            .vertex_count = vertex_count,
            .index_count = index_count,
            .index_type = comptime if (IndexType == u16) .uint16 else .uint32,
            .streams = upload.streams,
            .morph_target_count = morph_target_count,
            .bounds = res.Bounds.compute(vertices),
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

    // Where binding zero comes from when nothing overrides it.
    pub fn baseSource(self: *const Mesh) VertexSource {
        return .{ .handle = self.vertex_buffer.handle, .offset = 0 };
    }

    // Binds each present stream at its own binding. The optional ones sit at
    // fixed, non-contiguous bindings and appear independently, so they cannot be
    // bound as one range.
    //
    // `base` replaces binding zero and leaves every other stream alone, which is
    // what lets a morph prepass substitute its output without a second vertex
    // input or a second pipeline: the substitute has the same layout, because it
    // is a buffer of the same `GpuVertex`. Null is the mesh's own.
    pub fn bind(
        self: *const Mesh,
        context: *const Context,
        command_buffer: vk.CommandBuffer,
        base: ?VertexSource,
    ) void {
        const start: [1]vk.DeviceSize = .{0};
        const source = base orelse self.baseSource();
        context.device.cmdBindVertexBuffers(
            command_buffer,
            0,
            &.{source.handle},
            &.{source.offset},
        );
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

// Every buffer a mesh owns while it is being built, rolled back together.
//
// Destruction is skipped when the transfer's submission may still be pending.
// Vulkan specification, vkDestroyBuffer: submitted work reading a buffer must
// have completed, and a submission that failed after reaching the queue
// establishes neither completion nor that it never started. A leak is
// recoverable and a use after free is not.
//
// That case is why this is a struct and not six locals with six errdefers. A
// mesh is now filled through a pool that reclaims itself by submitting, so the
// question reaches every buffer at once and is answered in one place.
const Built = struct {
    vertex: ?Buffer = null,
    skin: ?Buffer = null,
    colour: ?Buffer = null,
    uv1: ?Buffer = null,
    morph: ?Buffer = null,
    index: ?Buffer = null,

    fn rollback(self: *Built, transfer: *const Transfer) void {
        if (transfer.abandoned()) {
            std.log.err(
                "mesh upload abandoned after a failed submission: its buffers " ++
                    "are leaked because the copies naming them may still be pending",
                .{},
            );
            return;
        }
        inline for (@typeInfo(Built).@"struct".fields) |field| {
            if (@field(self, field.name)) |*owned| owned.deinit();
        }
    }
};

// The device-local destination for one stream. Its size is the whole stream's;
// what arrives in pieces is the copying, not the allocation.
fn deviceBuffer(
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    comptime T: type,
    count: usize,
    usage: vk.BufferUsageFlags,
) buffer_module.InitError!Buffer {
    return Buffer.init(
        context,
        memory_allocator,
        @as(vk.DeviceSize, count) * @sizeOf(T),
        usage.merge(.{ .transfer_dst_bit = true }),
        .device,
    );
}

// Fills one device buffer from the staging pool, a chunk at a time. Whatever the
// pool can give is what a chunk is, so a stream larger than a staging block is a
// sequence of copies rather than a failure.
//
// The copy is recorded when the chunk is reserved, before the caller writes it.
// Nothing reaches the queue until a later reservation has to flush or the job
// finishes, and by then the chunk is full. The command buffer is read at the
// moment of recording rather than held across a reservation, because a flush
// retires it.
fn Chunks(comptime T: type) type {
    if (@alignOf(T) > stream_alignment)
        @compileError("a stream element aligned wider than a chunk cannot be written in place");

    return struct {
        const Self = @This();

        const Chunk = struct {
            items: []T,
            // Where the chunk starts in the stream, which is what the packing
            // loops index their source by.
            first: usize,
        };

        transfer: *Transfer,
        destination: *const Buffer,
        total: usize,
        written: usize,

        fn init(transfer: *Transfer, destination: *const Buffer, total: usize) Self {
            return .{
                .transfer = transfer,
                .destination = destination,
                .total = total,
                .written = 0,
            };
        }

        fn next(self: *Self) StreamError!?Chunk {
            if (self.written == self.total) return null;

            const reservation = try self.transfer.reserve(.{
                .size = @as(vk.DeviceSize, self.total - self.written) * @sizeOf(T),
                .alignment = stream_alignment,
                // A chunk is a whole number of elements, so no packing loop ever
                // writes across the seam between two of them.
                .granularity = @sizeOf(T),
            });

            // Exact: the pool returns either the whole request or a multiple of
            // the granularity, and both are multiples of the element size.
            const items: []T = @alignCast(std.mem.bytesAsSlice(T, reservation.bytes));
            const first = self.written;
            self.written += items.len;

            try self.destination.recordCopyFrom(
                reservation.source,
                self.transfer.commandBuffer(),
                .{
                    .source_offset = reservation.offset,
                    .destination_offset = @as(vk.DeviceSize, first) * @sizeOf(T),
                    .size = reservation.bytes.len,
                },
            );
            return .{ .items = items, .first = first };
        }
    };
}

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
