const std = @import("std");
const vk = @import("vulkan");
const buffer_module = @import("buffer.zig");
const Context = @import("context.zig").Context;
const descriptors = @import("descriptors.zig");
const memory = @import("memory/allocator.zig");
const mesh_module = @import("mesh/resource.zig");
const per_frame = @import("per_frame.zig");
const pipeline = @import("pipeline.zig");
const shaders = @import("shaders.zig");
const vertex_module = @import("mesh/vertex.zig");

const Allocator = std.mem.Allocator;
const Buffer = buffer_module.Buffer;
const GpuVertex = vertex_module.GpuVertex;
// What `Mesh.bind` takes as its override. The prepass produces one of these
// rather than declaring its own: a substitute for binding zero is the mesh's
// concept, and two spellings of it would be two things to keep in step.
const VertexSource = mesh_module.VertexSource;

// The morph prepass: a compute dispatch per morphed mesh, resolving its shape
// targets into a vertex buffer the main pass draws in place of the mesh's own.
//
// Why a prepass and not a loop in the vertex shader, which is what skinning
// does: a morphed vertex reads target_count times twenty-four bytes of
// displacement with no reuse, and a shader that resolves them pays that in every
// pass that draws the mesh. A skinned vertex instead reads four matrices out of
// an array small enough to stay resident. The two workloads differ in what the
// data does, so they are resolved in different places.
//
// The output is a whole GpuVertex, so nothing downstream is aware of it: the
// recorder binds this buffer at binding 0 instead of the mesh's, and both the
// vertex input and scene.slang are unchanged. A mesh that is also skinned skins
// in the vertex stage over these vertices.
//
// Mirrors assets/shaders/morph.slang. `tests/reflection.zig` holds the bindings,
// the push block and the workgroup size against the compiler's account of it.

// One group of threads covers this many vertices. It is declared in the shader,
// where Vulkan reads it from as an execution mode, and repeated here because the
// dispatch has to divide by the same number. Reflection is what holds the two
// together rather than this comment.
pub const group_size: u32 = 64;

pub const bindings = [_]descriptors.Binding{
    // The mesh's own packed vertices, read and not written.
    .{ .slot = 0, .kind = .storage_buffer, .stages = .{ .compute_bit = true } },
    // Its displacements, one element per (vertex, target) pair.
    .{ .slot = 1, .kind = .storage_buffer, .stages = .{ .compute_bit = true } },
    // The frame's weights. Dynamic, because it is one ring over every frame in
    // flight and the slot is chosen at bind time, exactly as the joint array is.
    .{ .slot = 2, .kind = .storage_buffer_dynamic, .stages = .{ .compute_bit = true } },
    // The destination. Not dynamic: it is per mesh and device-local rather than
    // a host-visible ring, so the frame it writes is a push constant instead.
    .{ .slot = 3, .kind = .storage_buffer, .stages = .{ .compute_bit = true } },
};

const Sets = descriptors.Sets(&bindings);

// Vulkan specification, vkCmdBindDescriptorSets: one dynamic offset per dynamic
// descriptor, consumed in increasing binding order. Derived from the table rather
// than written beside the bind, so a second dynamic binding stops the build in
// `record` instead of handing the weights someone else's offset.
const dynamic_count = blk: {
    var count = 0;
    for (bindings) |binding| {
        switch (binding.kind) {
            .storage_buffer_dynamic, .uniform_buffer_dynamic => count += 1,
            else => {},
        }
    }
    break :blk count;
};

// What one dispatch needs that is not in its descriptor set.
pub const PushConstants = extern struct {
    vertex_count: u32,
    target_count: u32,
    // Where this mesh's weights begin inside the frame's slot. The slot itself
    // is the dynamic offset.
    weight_base: u32,
    // Where this frame's vertices begin in the destination, in elements.
    output_base: u32,
};

pub const push_constant_range: vk.PushConstantRange = .{
    .stage_flags = .{ .compute_bit = true },
    .offset = 0,
    .size = @sizeOf(PushConstants),
};

pub const InitError = descriptors.InitError ||
    per_frame.InitError ||
    per_frame.LayoutError ||
    pipeline.CreateError;

pub const RegisterError = error{
    // More morphed meshes than the pass was built for. A runtime condition: how
    // many a model carries is the model's business.
    MeshCapacityExceeded,

    // More weights across every registered mesh than one frame's slot holds.
    WeightCapacityExceeded,

    // A mesh with no displacements. Nothing here can resolve it, and a
    // registration that silently did nothing would draw the bind shape and
    // report success.
    NotMorphed,

    // The destination would hold more elements than an index into it can
    // address. The mesh itself is bounded by a u32 vertex count, and this is
    // that count times the frames in flight.
    DestinationTooLarge,
} || buffer_module.InitError;

pub const WriteError = error{
    // Weights of a length the registered mesh does not have. The shader reads
    // exactly target_count of them, so a short array leaves the rest as another
    // mesh left them.
    WeightCountMismatch,
    RegistrationOutOfRange,
};

// How much the pass holds. Both bounds are the caller's, because how many
// morphed meshes a scene has is not the pass's to decide.
pub const Capacity = struct {
    // Morphed meshes. One descriptor set and one destination buffer each.
    meshes: u32,
    // Weights in one frame's slot, summed over every registered mesh.
    weights: usize,
};

// How many groups cover a mesh, and the only arithmetic in this file that a test
// can reach without a device.
//
// Rounded up, so the last group runs with threads that address no vertex; the
// shader's own bounds test is what makes those threads write nothing. Rounding
// down instead leaves the tail of every mesh whose count is not a multiple of the
// group size at its bind shape, which looks like a partly animated mesh.
//
// std.math.divCeil returns DivisionByZero for a zero denominator and Overflow
// only for a signed minInt over minus one. Read in the installed std source of
// both 0.16 and 0.17-dev, whose implementations differ and whose error set does
// not: with an unsigned numerator and the nonzero comptime denominator the assert
// below pins, neither is reachable.
pub fn groupsFor(vertex_count: u32) u32 {
    return std.math.divCeil(u32, vertex_count, group_size) catch unreachable;
}

comptime {
    std.debug.assert(group_size > 0);
}

// Where one registration's weights go, and what the running total becomes.
//
// Split from `register` for the reason `frame_set.validate` is split from the
// write it guards: everything a registration can get wrong about capacity is
// arithmetic, and a device is what stands between that arithmetic and a test.
pub const Reservation = struct {
    base: u32,
    used: usize,
};

pub fn reserveWeights(capacity: usize, used: usize, target_count: u32) RegisterError!Reservation {
    // The subtraction cannot go negative: `used` never exceeds `capacity`,
    // because this call is the only thing that advances it and it refuses to
    // advance past it.
    if (target_count > capacity - used) return error.WeightCapacityExceeded;
    return .{ .base = @intCast(used), .used = used + target_count };
}

// Whether one more mesh fits. The bound is the descriptor set count, so the two
// things a registration consumes are one number.
pub fn validateRegistration(registered: usize, capacity: usize) RegisterError!void {
    if (registered >= capacity) return error.MeshCapacityExceeded;
}

// The shader reads exactly `target_count` weights, so a short array leaves the
// rest as whichever mesh wrote that part of the slot last.
pub fn validateWeightCount(target_count: u32, count: usize) WriteError!void {
    if (count != target_count) return error.WeightCountMismatch;
}

// How many vertices a destination holds, or an error when an index into it would
// not fit the push constant that carries one.
pub fn destinationElements(frames: usize, vertex_count: u32) RegisterError!u32 {
    const total = std.math.mul(usize, frames, vertex_count) catch
        return error.DestinationTooLarge;
    if (total > std.math.maxInt(u32)) return error.DestinationTooLarge;
    return @intCast(total);
}

// One registered mesh. The source buffers are borrowed: the mesh outlives the
// registration, which `register` states as its precondition. The destination is
// owned, because it is per instance and per frame and the mesh is neither.
const Entry = struct {
    source: vk.Buffer,
    deltas: vk.Buffer,
    destination: Buffer,
    vertex_count: u32,
    target_count: u32,
    weight_base: u32,
    groups: u32,
};

pub const MorphPass = struct {
    context: *const Context,
    allocator: Allocator,
    frames: usize,

    module: vk.ShaderModule,
    layout: vk.PipelineLayout,
    pipeline: vk.Pipeline,

    sets: Sets,
    // One ring over every frame in flight, holding every registered mesh's
    // weights end to end. One array rather than one per mesh, for the reason the
    // joint array is one per frame: a dispatch reaches its own run through the
    // base it carries.
    weights: per_frame.PerFrame(f32),
    entries: std.ArrayList(Entry),
    // The running total that lays the weights out. Registration is the only
    // thing that moves it, so no separate planning pass exists.
    weights_used: usize,

    pub fn init(
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        allocator: Allocator,
        frames: usize,
        capacity: Capacity,
    ) InitError!MorphPass {
        var sets = try Sets.init(context, allocator, capacity.meshes);
        errdefer sets.deinit(context, allocator);

        var weights = try per_frame.PerFrame(f32).init(
            context,
            memory_allocator,
            frames,
            capacity.weights,
            .{ .storage_buffer_bit = true },
        );
        errdefer weights.deinit();

        const module = try pipeline.createModule(context, shaders.morph.spirv);
        errdefer context.device.destroyShaderModule(module, null);

        const layout = try pipeline.createLayout(context, .{
            .descriptor_sets = &.{sets.layout},
            .push_constants = &.{push_constant_range},
        });
        errdefer context.device.destroyPipelineLayout(layout, null);

        const built = try pipeline.createCompute(context, .{
            .layout = layout,
            .stage = .{
                .module = module,
                .entry_point = shaders.morph.compute_entry.?,
            },
        });
        errdefer context.device.destroyPipeline(built, null);

        // Fixed at the set count, so the two bounds registration answers to are
        // one number rather than two that can drift.
        const entries = try std.ArrayList(Entry).initCapacity(allocator, capacity.meshes);

        return .{
            .context = context,
            .allocator = allocator,
            .frames = frames,
            .module = module,
            .layout = layout,
            .pipeline = built,
            .sets = sets,
            .weights = weights,
            .entries = entries,
            .weights_used = 0,
        };
    }

    // Vulkan specification, vkDestroyPipeline and the rest: every submission
    // naming any of these must have completed. The caller drains the device.
    pub fn deinit(self: *MorphPass) void {
        const device = self.context.device;
        for (self.entries.items) |*entry| entry.destination.deinit();
        self.entries.deinit(self.allocator);
        self.weights.deinit();
        self.sets.deinit(self.context, self.allocator);
        device.destroyPipeline(self.pipeline, null);
        device.destroyPipelineLayout(self.layout, null);
        device.destroyShaderModule(self.module, null);
        self.* = undefined;
    }

    // Take a morphed mesh into the pass, allocating its destination and writing
    // the descriptor set that reads it.
    //
    // The mesh is borrowed and must outlive the pass, because the set holds its
    // two buffers. Cold: once per morphed mesh when the model becomes resident.
    //
    // The weights are laid out here, end to end in registration order, so the
    // returned index is both the entry and the descriptor set.
    pub fn register(
        self: *MorphPass,
        memory_allocator: *memory.MemoryAllocator,
        mesh: *const mesh_module.Mesh,
    ) RegisterError!u32 {
        try validateRegistration(self.entries.items.len, self.sets.sets.len);
        const deltas = mesh.morph_buffer orelse return error.NotMorphed;
        if (mesh.morph_target_count == 0) return error.NotMorphed;

        const reserved = try reserveWeights(
            self.weights.count,
            self.weights_used,
            mesh.morph_target_count,
        );
        const elements = try destinationElements(self.frames, mesh.vertex_count);
        var destination = try Buffer.init(
            self.context,
            memory_allocator,
            @as(vk.DeviceSize, elements) * @sizeOf(GpuVertex),
            .{ .storage_buffer_bit = true, .vertex_buffer_bit = true },
            .device,
        );
        errdefer destination.deinit();

        const index: u32 = @intCast(self.entries.items.len);

        // Written before the entry is appended, so a failure above leaves the
        // set pointing at nothing rather than a registration nothing dispatches.
        // The infos are named because the write holds a pointer to each.
        const infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = mesh.vertex_buffer.handle, .offset = 0, .range = vk.WHOLE_SIZE },
            .{ .buffer = deltas.handle, .offset = 0, .range = vk.WHOLE_SIZE },
            self.weights.descriptor(),
            .{ .buffer = destination.handle, .offset = 0, .range = vk.WHOLE_SIZE },
        };
        var writes: [bindings.len]vk.WriteDescriptorSet = undefined;
        for (bindings, &infos, &writes) |binding, *info, *write| {
            write.* = .{
                .dst_set = self.sets.set(index),
                .dst_binding = binding.slot,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = binding.kind,
                .p_image_info = &no_images,
                .p_buffer_info = @ptrCast(info),
                .p_texel_buffer_view = &no_texel_buffers,
            };
        }
        self.context.device.updateDescriptorSets(&writes, null);

        self.entries.appendAssumeCapacity(.{
            .source = mesh.vertex_buffer.handle,
            .deltas = deltas.handle,
            .destination = destination,
            .vertex_count = mesh.vertex_count,
            .target_count = mesh.morph_target_count,
            .weight_base = reserved.base,
            .groups = groupsFor(mesh.vertex_count),
        });
        self.weights_used = reserved.used;
        return index;
    }

    pub fn registrationCount(self: *const MorphPass) usize {
        return self.entries.items.len;
    }

    // Fill one registration's weights in one frame's slot.
    //
    // The caller has waited on that slot's fence; see the invariant on
    // `PerFrame`. Nothing here is a device call.
    pub fn writeWeights(
        self: *MorphPass,
        frame: usize,
        registration: u32,
        values: []const f32,
    ) WriteError!void {
        if (registration >= self.entries.items.len) return error.RegistrationOutOfRange;
        const entry = &self.entries.items[registration];
        try validateWeightCount(entry.target_count, values.len);

        const slot = self.weights.slice(frame);
        @memcpy(slot[entry.weight_base..][0..values.len], values);
    }

    // Where the recorder fetches a registration's vertices for this frame.
    pub fn vertexSource(self: *const MorphPass, registration: u32, frame: usize) VertexSource {
        std.debug.assert(registration < self.entries.items.len);
        std.debug.assert(frame < self.frames);
        const entry = &self.entries.items[registration];
        return .{
            .handle = entry.destination.handle,
            .offset = @as(vk.DeviceSize, frame) * entry.vertex_count * @sizeOf(GpuVertex),
        };
    }

    // Every registered mesh, dispatched, followed by the one barrier that makes
    // the results readable as vertex attributes.
    //
    // There is no barrier in front of the dispatches. The destination slot this
    // frame writes was last read by the submission this frame's fence stands
    // for, and the caller has waited on it, so the hazard is a write after a
    // read that has already completed. That needs execution ordering and nothing
    // else, which the fence wait is.
    pub fn record(self: *const MorphPass, command_buffer: vk.CommandBuffer, frame: usize) void {
        if (self.entries.items.len == 0) return;
        std.debug.assert(frame < self.frames);

        const device = self.context.device;
        device.cmdBindPipeline(command_buffer, .compute, self.pipeline);

        const offsets: [dynamic_count]u32 = .{self.weights.dynamicOffset(frame)};
        for (self.entries.items, 0..) |*entry, index| {
            device.cmdBindDescriptorSets(
                command_buffer,
                .compute,
                self.layout,
                0,
                &.{self.sets.set(index)},
                &offsets,
            );
            const push: PushConstants = .{
                .vertex_count = entry.vertex_count,
                .target_count = entry.target_count,
                .weight_base = entry.weight_base,
                // Bounded by destinationElements at registration.
                .output_base = @intCast(frame * entry.vertex_count),
            };
            device.cmdPushConstants(
                command_buffer,
                self.layout,
                push_constant_range.stage_flags,
                push_constant_range.offset,
                push_constant_range.size,
                @ptrCast(&push),
            );
            device.cmdDispatch(command_buffer, entry.groups, 1, 1);
        }

        const barriers = [_]vk.MemoryBarrier2{barrier()};
        device.cmdPipelineBarrier2(command_buffer, &.{
            .memory_barrier_count = barriers.len,
            .p_memory_barriers = &barriers,
        });
    }

    // The layout a compute pipeline for this pass is built from, exposed for the
    // same reason the frame set exposes its own.
    pub fn descriptorSetLayout(self: *const MorphPass) vk.DescriptorSetLayout {
        return self.sets.layout;
    }
};

// What the prepass wrote, made available to the vertex fetch that reads it.
//
// One memory barrier and not one per destination: the dependency is the same for
// every buffer this pass touched, and stating it once is both fewer barriers and
// one place to be wrong. A buffer barrier would additionally have to name a
// range, which for a device-local destination buys no filtering the driver acts
// on here.
//
// The destination is read as a vertex attribute and not in a shader, so the
// destination scope is the vertex input stage. Naming `vertex_shader_bit`
// instead is the mistake this function exists to make once: the fetch happens
// before the shader runs, and the barrier would order nothing.
pub fn barrier() vk.MemoryBarrier2 {
    return .{
        .src_stage_mask = .{ .compute_shader_bit = true },
        .src_access_mask = .{ .shader_storage_write_bit = true },
        .dst_stage_mask = .{ .vertex_attribute_input_bit = true },
        .dst_access_mask = .{ .vertex_attribute_read_bit = true },
    };
}

// Vulkan specification, VkWriteDescriptorSet: the members not selected by
// descriptorType are ignored, but the pointers are not optional.
const no_images = [_]vk.DescriptorImageInfo{};
const no_texel_buffers = [_]vk.BufferView{};
