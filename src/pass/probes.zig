const std = @import("std");
const vk = @import("vulkan");

const acceleration = @import("../object/acceleration.zig");
const buffer_module = @import("../object/buffer.zig");
const Buffer = buffer_module.Buffer;
const commands = @import("../device/commands.zig");
const Context = @import("../device/context.zig").Context;
const descriptors = @import("../binding/descriptors.zig");
const memory = @import("../memory/allocator.zig");
const pipeline = @import("../binding/pipeline.zig");
const uniforms = @import("../binding/uniforms.zig");

const Allocator = std.mem.Allocator;

// The indirect-light probe field: a lattice of probes, each holding the
// second-order spherical harmonics of the radiance around it and a small
// octahedral map of the distances its rays travelled, rebuilt by a compute
// pass that traces the scene with inline ray queries. The main pass reads the
// two buffers through the scene set and lights every surface's diffuse
// indirect term from the eight probes around it.
//
// This owns the field's buffers (two of each, read one and write the other),
// the world-space triangles with their albedo, the acceleration structures
// over them, and the update pipeline. Settling is synchronous and belongs to
// load time: `settle` records its updates into a one-shot command buffer and
// waits. Updates spread over frames are not here yet.

pub const sh_count = 9;
pub const moment_texels = 64;
// One workgroup per probe; the shader's `numthreads`.
const rays_per_probe = 64;

// The grid as both shaders read it; defined beside the light block, which
// carries it to the main pass.
pub const GridParams = uniforms.ProbeGrid;

// Mirrors `Triangle` in probes.slang: world-space corners, the albedo's red,
// green and blue in the fourth lanes of the three.
pub const Triangle = extern struct {
    a: [4]f32,
    b: [4]f32,
    c: [4]f32,
};

// Mirrors `UpdatePushConstants` in probes.slang.
pub const PushConstants = extern struct {
    grid: GridParams,
    sky: [3]f32,
    light_count: u32,
    ground: [3]f32,
    hysteresis: f32,
    rotation: [4]f32,
    first_probe: u32,
    probe_count: u32,
    max_distance: f32,
    padding: f32 = 0,
};

pub const push_constant_range: vk.PushConstantRange = .{
    .stage_flags = .{ .compute_bit = true },
    .offset = 0,
    .size = @sizeOf(PushConstants),
};

pub const bindings = [_]descriptors.Binding{
    .{ .slot = 0, .name = "scene_acceleration", .kind = .acceleration_structure_khr, .stages = .{ .compute_bit = true } },
    .{ .slot = 1, .name = "triangles", .kind = .storage_buffer, .stages = .{ .compute_bit = true } },
    .{ .slot = 2, .name = "probe_lights", .kind = .storage_buffer, .stages = .{ .compute_bit = true } },
    .{ .slot = 3, .name = "probe_sh", .kind = .storage_buffer, .stages = .{ .compute_bit = true } },
    .{ .slot = 4, .name = "probe_moments", .kind = .storage_buffer, .stages = .{ .compute_bit = true } },
    .{ .slot = 5, .name = "next_sh", .kind = .storage_buffer, .stages = .{ .compute_bit = true } },
    .{ .slot = 6, .name = "next_moments", .kind = .storage_buffer, .stages = .{ .compute_bit = true } },
};

pub const Sets = descriptors.Sets(&bindings);

// The field's two buffers in the scene set, after the occlusion result's slot.
// The renderer joins these with the others there.
pub const scene_bindings = [_]descriptors.Binding{
    .{ .slot = 7, .name = "probe_sh", .kind = .storage_buffer, .stages = .{ .fragment_bit = true } },
    .{ .slot = 8, .name = "probe_moments", .kind = .storage_buffer, .stages = .{ .fragment_bit = true } },
};

pub const Shader = struct {
    spirv: []const u32,
    update_entry: [*:0]const u8,
};

// Points the scene set's two slots at a field's buffers, or at any storage
// buffer when there is none: the main pass reads them only while the grid in
// the light block is enabled, but a descriptor has to name something either
// way. When no frame is in flight.
pub fn writeScene(context: *const Context, scene_set: vk.DescriptorSet, sh_buffer: *const Buffer, moment_buffer: *const Buffer) void {
    const infos = [_]vk.DescriptorBufferInfo{
        .{ .buffer = sh_buffer.handle, .offset = 0, .range = vk.WHOLE_SIZE },
        .{ .buffer = moment_buffer.handle, .offset = 0, .range = vk.WHOLE_SIZE },
    };
    var writes: [scene_bindings.len]vk.WriteDescriptorSet = undefined;
    for (&writes, scene_bindings, &infos) |*write, binding, *info| {
        write.* = .{
            .dst_set = scene_set,
            .dst_binding = binding.slot,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = binding.kind,
            .p_image_info = &no_images,
            .p_buffer_info = @ptrCast(info),
            .p_texel_buffer_view = &no_texel_buffers,
        };
    }
    context.device.updateDescriptorSets(&writes, null);
}

// Vulkan specification, VkWriteDescriptorSet: the members not selected by
// descriptorType are ignored, but the pointers are not optional.
const no_images = [_]vk.DescriptorImageInfo{};
const no_texel_buffers = [_]vk.BufferView{};

pub const InitError = error{
    // The device did not enable ray queries, which the update traces with.
    RayQueryDisabled,
    // No triangles to trace, or no probes.
    EmptyField,
    // More triangles than a primitive index or a buffer size can carry.
    TooManyTriangles,
} || buffer_module.InitError || buffer_module.UploadError || acceleration.BuildError ||
    commands.BeginError || commands.SubmitError || descriptors.InitError ||
    descriptors.WriteError || pipeline.CreateError;

pub const SettleError = error{
    // More lights than the update's buffer holds.
    TooManyLights,
} || buffer_module.UploadError || commands.BeginError || commands.SubmitError;

// What an update lights the field with, besides the scene.
pub const Lighting = struct {
    lights: []const uniforms.Light,
    // Radiance from where a ray leaves the scene: the ambient's sky and ground.
    sky: [3]f32 = .{ 0, 0, 0 },
    ground: [3]f32 = .{ 0, 0, 0 },
    // How far a ray looks before it counts as leaving the scene, in metres.
    max_distance: f32 = 100,
};

// The rotation applied to the rays of update `index`, as a unit quaternion
// (x, y, z, w). Successive updates turn the ray set by the golden angle about
// axes that themselves walk a Fibonacci spiral, so the blended field converges
// over the sphere rather than over the same 64 directions; any well-spread
// sequence serves, and this one needs no random state.
pub fn rotationFor(index: u32) [4]f32 {
    const golden_angle: f32 = std.math.pi * (3.0 - @sqrt(5.0));
    const n: f32 = 97;
    const i: f32 = @floatFromInt(index % 97);
    const y = 1 - (2 * i + 1) / n;
    const radius = @sqrt(@max(0, 1 - y * y));
    const axis_angle = i * golden_angle;
    const axis = [3]f32{ radius * @cos(axis_angle), y, radius * @sin(axis_angle) };
    const half = 0.5 * @as(f32, @floatFromInt(index)) * golden_angle;
    const s = @sin(half);
    return .{ axis[0] * s, axis[1] * s, axis[2] * s, @cos(half) };
}

pub const ProbeField = struct {
    context: *const Context,
    allocator: Allocator,
    module: vk.ShaderModule,
    layout: vk.PipelineLayout,
    pipeline: vk.Pipeline,
    // Set `i` reads pair `i` and writes pair `1 - i`.
    sets: Sets,
    grid: GridParams,
    sh: [2]Buffer,
    moments: [2]Buffer,
    triangles: Buffer,
    lights: Buffer,
    bottom_level: acceleration.AccelerationStructure,
    top_level: acceleration.AccelerationStructure,
    // The pair holding the settled field, which is what the main pass reads.
    current: u1,

    // Builds the acceleration structures over `world_triangles` and clears the
    // field. Synchronous: records into a one-shot command buffer from `pool`
    // and waits, which is a load-time cost.
    pub fn init(
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        allocator: Allocator,
        pool: vk.CommandPool,
        grid: GridParams,
        world_triangles: []const Triangle,
        shader: Shader,
    ) InitError!ProbeField {
        if (!context.ray_query_enabled) return error.RayQueryDisabled;
        const probe_count = grid.probeCount();
        if (probe_count == 0 or world_triangles.len == 0) return error.EmptyField;
        // Three vertices each, counted in a u32 by the bottom-level build.
        if (world_triangles.len > std.math.maxInt(u32) / 3) return error.TooManyTriangles;

        const storage: vk.BufferUsageFlags = .{ .storage_buffer_bit = true, .transfer_dst_bit = true };
        var sh: [2]Buffer = undefined;
        var moments: [2]Buffer = undefined;
        var created: usize = 0;
        errdefer for (0..created) |i| {
            sh[i].deinit();
            moments[i].deinit();
        };
        for (0..2) |i| {
            sh[i] = try Buffer.init(context, memory_allocator, @as(u64, probe_count) * sh_count * 16, storage, .device);
            moments[i] = Buffer.init(context, memory_allocator, @as(u64, probe_count) * moment_texels * 8, storage, .device) catch |err| {
                sh[i].deinit();
                return err;
            };
            created += 1;
        }

        var triangles = try Buffer.init(
            context,
            memory_allocator,
            @as(u64, world_triangles.len) * @sizeOf(Triangle),
            .{ .storage_buffer_bit = true },
            .upload,
        );
        errdefer triangles.deinit();
        try triangles.upload(std.mem.sliceAsBytes(world_triangles));

        var lights = try Buffer.init(
            context,
            memory_allocator,
            @as(u64, uniforms.max_lights) * @sizeOf(uniforms.Light),
            .{ .storage_buffer_bit = true },
            .upload,
        );
        errdefer lights.deinit();

        // The positions alone, three vertices a triangle, which is what the
        // bottom-level build reads. Freed once the build has run.
        var positions = try Buffer.init(
            context,
            memory_allocator,
            @as(u64, world_triangles.len) * 3 * @sizeOf([3]f32),
            .{ .shader_device_address_bit = true, .acceleration_structure_build_input_read_only_bit_khr = true },
            .upload,
        );
        defer positions.deinit();
        {
            const mapped = positions.allocation.mappedBytes() orelse return error.BufferNotHostVisible;
            const out = std.mem.bytesAsSlice([3]f32, mapped[0 .. world_triangles.len * 3 * @sizeOf([3]f32)]);
            for (world_triangles, 0..) |triangle, i| {
                out[3 * i] = triangle.a[0..3].*;
                out[3 * i + 1] = triangle.b[0..3].*;
                out[3 * i + 2] = triangle.c[0..3].*;
            }
        }

        const command = try commands.beginOneShot(context, pool);
        errdefer context.device.freeCommandBuffers(pool, &.{command});
        var bottom_pending = try acceleration.recordBottomLevel(context, memory_allocator, command, .{
            .positions = &positions,
            .vertex_count = @intCast(world_triangles.len * 3),
        });
        errdefer bottom_pending.deinit();
        var top_pending = try acceleration.recordTopLevel(context, memory_allocator, command, &.{
            .{ .bottom_level = &bottom_pending.structure },
        });
        errdefer top_pending.deinit();
        // A field that has never been updated reads as no light and no
        // visibility: zero harmonics, zero moments.
        for (0..2) |i| {
            context.device.cmdFillBuffer(command, sh[i].handle, 0, vk.WHOLE_SIZE, 0);
            context.device.cmdFillBuffer(command, moments[i].handle, 0, vk.WHOLE_SIZE, 0);
        }
        recordBarrier(context, command, .{ .clear_bit = true }, .{ .transfer_write_bit = true });
        try commands.submitOneShotAndWait(context, pool, command);
        var bottom_level = bottom_pending.finish();
        errdefer bottom_level.deinit();
        var top_level = top_pending.finish();
        errdefer top_level.deinit();

        const module = try pipeline.createModule(context, shader.spirv);
        errdefer context.device.destroyShaderModule(module, null);
        var sets = try Sets.init(context, allocator, 2);
        errdefer sets.deinit(context, allocator);
        const layout = try pipeline.createLayout(context, .{
            .descriptor_sets = &.{sets.layout},
            .push_constants = &.{push_constant_range},
        });
        errdefer context.device.destroyPipelineLayout(layout, null);
        const update_pipeline = try pipeline.createCompute(context, .{
            .layout = layout,
            .stage = .{ .module = module, .entry_point = shader.update_entry },
        });
        errdefer context.device.destroyPipeline(update_pipeline, null);

        for (0..2) |i| {
            const other = 1 - i;
            sets.writeBuffers(context, i, .{
                .triangles = descriptors.BufferSource{ .buffer = &triangles },
                .probe_lights = descriptors.BufferSource{ .buffer = &lights },
                .probe_sh = descriptors.BufferSource{ .buffer = &sh[i] },
                .probe_moments = descriptors.BufferSource{ .buffer = &moments[i] },
                .next_sh = descriptors.BufferSource{ .buffer = &sh[other] },
                .next_moments = descriptors.BufferSource{ .buffer = &moments[other] },
            });
            try sets.writeAccelerationStructures(context, i, .{
                .scene_acceleration = descriptors.AccelerationSource.one(&top_level),
            });
        }

        return .{
            .context = context,
            .allocator = allocator,
            .module = module,
            .layout = layout,
            .pipeline = update_pipeline,
            .sets = sets,
            .grid = grid,
            .sh = sh,
            .moments = moments,
            .triangles = triangles,
            .lights = lights,
            .bottom_level = bottom_level,
            .top_level = top_level,
            .current = 0,
        };
    }

    // Vulkan specification, vkDestroy*: every submission naming any of these
    // must have completed. The caller drains the device.
    pub fn deinit(self: *ProbeField) void {
        const device = self.context.device;
        device.destroyPipeline(self.pipeline, null);
        device.destroyPipelineLayout(self.layout, null);
        self.sets.deinit(self.context, self.allocator);
        device.destroyShaderModule(self.module, null);
        self.top_level.deinit();
        self.bottom_level.deinit();
        self.lights.deinit();
        self.triangles.deinit();
        for (&self.sh) |*buffer| buffer.deinit();
        for (&self.moments) |*buffer| buffer.deinit();
        self.* = undefined;
    }

    // Runs `iterations` updates of every probe and waits. The first replaces
    // whatever the field held; each after it blends in with `hysteresis`, and
    // each carries the light one bounce further through the field it reads.
    // Leaves the field in the pair `current` names, its writes visible to the
    // fragment stage.
    pub fn settle(
        self: *ProbeField,
        pool: vk.CommandPool,
        lighting: Lighting,
        iterations: u32,
        hysteresis: f32,
    ) SettleError!void {
        if (lighting.lights.len > uniforms.max_lights) return error.TooManyLights;
        try self.lights.upload(std.mem.sliceAsBytes(lighting.lights));

        const device = self.context.device;
        const command = try commands.beginOneShot(self.context, pool);
        errdefer device.freeCommandBuffers(pool, &.{command});
        device.cmdBindPipeline(command, .compute, self.pipeline);

        const probe_count = self.grid.probeCount();
        var iteration: u32 = 0;
        while (iteration < iterations) : (iteration += 1) {
            device.cmdBindDescriptorSets(command, .compute, self.layout, 0, &.{self.sets.set(self.current)}, &.{});
            var push: PushConstants = .{
                .grid = self.grid,
                .sky = lighting.sky,
                .light_count = @intCast(lighting.lights.len),
                .ground = lighting.ground,
                .hysteresis = settleHysteresis(iteration, hysteresis),
                .rotation = rotationFor(iteration),
                .first_probe = 0,
                .probe_count = probe_count,
                .max_distance = lighting.max_distance,
            };
            device.cmdPushConstants(command, self.layout, push_constant_range.stage_flags, 0, @sizeOf(PushConstants), @ptrCast(&push));
            device.cmdDispatch(command, probe_count, 1, 1);
            // The next update reads what this one wrote.
            recordBarrier(self.context, command, .{ .compute_shader_bit = true }, .{ .shader_storage_write_bit = true });
            self.current = 1 - self.current;
        }
        // The main pass reads the settled pair. A queue wait follows, so this
        // is for the record rather than for ordering; it names the reader.
        const done = [_]vk.MemoryBarrier2{.{
            .src_stage_mask = .{ .compute_shader_bit = true },
            .src_access_mask = .{ .shader_storage_write_bit = true },
            .dst_stage_mask = .{ .fragment_shader_bit = true },
            .dst_access_mask = .{ .shader_storage_read_bit = true },
        }};
        device.cmdPipelineBarrier2(command, &.{ .memory_barrier_count = done.len, .p_memory_barriers = &done });
        try commands.submitOneShotAndWait(self.context, pool, command);
    }

    // What the main pass reads: the settled pair.
    pub fn shBuffer(self: *const ProbeField) *const Buffer {
        return &self.sh[self.current];
    }

    pub fn momentBuffer(self: *const ProbeField) *const Buffer {
        return &self.moments[self.current];
    }
};

// How much of the field survives settle update `iteration`: k / (k + 1), which
// makes the first updates a plain average of every ray so far, capped at
// `limit`, past which each update keeps a fixed share. The cap is what lets the
// dark first bounces wash out: an average over every update would keep them, at
// one part in the update count, while the light they lacked arrives.
pub fn settleHysteresis(iteration: u32, limit: f32) f32 {
    const k: f32 = @floatFromInt(iteration);
    return @min(k / (k + 1), limit);
}

// Writes of `stage` and `access` before compute reads and writes of the field.
fn recordBarrier(
    context: *const Context,
    command: vk.CommandBuffer,
    stage: vk.PipelineStageFlags2,
    access: vk.AccessFlags2,
) void {
    const barriers = [_]vk.MemoryBarrier2{.{
        .src_stage_mask = stage,
        .src_access_mask = access,
        .dst_stage_mask = .{ .compute_shader_bit = true },
        .dst_access_mask = .{ .shader_storage_read_bit = true, .shader_storage_write_bit = true },
    }};
    context.device.cmdPipelineBarrier2(command, &.{ .memory_barrier_count = barriers.len, .p_memory_barriers = &barriers });
}

comptime {
    // The shader's group size and the rays an update counts on.
    std.debug.assert(rays_per_probe == moment_texels);
}
