const std = @import("std");
const vk = @import("vulkan");
const commands = @import("../device/commands.zig");
const Context = @import("../device/context.zig").Context;
const descriptors = @import("../binding/descriptors.zig");
const memory = @import("../memory/allocator.zig");
const pipeline = @import("../binding/pipeline.zig");
const Image = @import("../object/image.zig").Image;
const Buffer = @import("../object/buffer.zig").Buffer;
const Bound = @import("../object/texture_cache.zig").Bound;

// The physical sky, rendered into a cube that becomes the scene's GGX
// environment. The background draws that cube and the surfaces reflect it, so
// the sky and its reflections are one image, as they are with an environment
// loaded from files.
//
// The transport tables are not computed here. The host builds them once, in
// `lenore-scene`'s atmosphere, and they arrive as two bound textures: one
// source for both halves, so this pass cannot drift from the model the host
// lights the scene with. What runs on the device is the march along each
// texel's direction (Hillaire 2020, Epic's reference `IntegrateScatteredLuminance`
// as `SkyViewLutPS` calls it), and only when the sun has moved.
//
// The mips are linear blits of level zero. The scene samples them by roughness,
// so a rough surface reflects a box-filtered sky rather than a GGX-filtered
// one; a proper prefilter is not written.

// Texels along a face's side at level zero, and the levels down to one texel.
// About 0.7 degrees a texel, which holds the sky's gradient; the sun's disc and
// halo are the background's own, drawn analytically over it.
pub const extent: u32 = 128;
pub const levels: u32 = 8;

// The shader's `Atmosphere`, as rows of four floats so the layout is the
// same in both languages with no padding to keep in step. Composition fills
// it from the host model; this module does not depend on `lenore-scene`.
//
//   0  bottom radius, top radius, Rayleigh scale height, Mie scale height
//   1  Rayleigh scattering, unused
//   2  Mie scattering, unused
//   3  Mie extinction, unused
//   4  ozone absorption, unused
//   5  ozone layer width, lower slope, lower intercept, upper slope
//   6  upper intercept, aerosol, ozone, Mie g
//   7  least and most samples, the viewer's altitude, unused
//   8  ground albedo, unused
pub const Parameters = extern struct { rows: [9][4]f32 };

// What changes with the time of day: the direction toward the sun and its
// irradiance above the atmosphere. Pushed, since it is all a relight changes.
pub const Sun = extern struct { toward_sun: [3]f32, top: f32 };

pub const Shader = struct { spirv: []const u32, compute_entry: [*:0]const u8 };

pub const bindings = [_]descriptors.Binding{
    .{ .slot = 0, .name = "transmittance", .kind = .combined_image_sampler, .stages = .{ .compute_bit = true } },
    .{ .slot = 1, .name = "multiple", .kind = .combined_image_sampler, .stages = .{ .compute_bit = true } },
    .{ .slot = 2, .name = "model", .kind = .storage_buffer, .stages = .{ .compute_bit = true } },
    .{ .slot = 3, .name = "cube", .kind = .storage_image, .stages = .{ .compute_bit = true } },
};
const Sets = descriptors.Sets(&bindings);

const push_range: vk.PushConstantRange = .{
    .stage_flags = .{ .compute_bit = true },
    .offset = 0,
    .size = @sizeOf(Sun),
};

// The direction a point on a cube face stands for. Vulkan specification, Cube
// Map Face Selection and Transformations: its table gives the face's major
// axis and the s and t it is sampled at, and this inverts it. `uv` is in
// [0, 1] across the face, corners included; a texel's centre is
// (xy + 0.5) / extent. The shader's `cubeDirection` is the same table.
pub fn cubeDirection(face: u32, uv: [2]f32) error{InvalidCubeCoordinate}![3]f32 {
    if (face >= 6 or !std.math.isFinite(uv[0]) or !std.math.isFinite(uv[1]) or
        uv[0] < 0 or uv[0] > 1 or uv[1] < 0 or uv[1] > 1) return error.InvalidCubeCoordinate;
    const s = 2 * uv[0] - 1;
    const t = 2 * uv[1] - 1;
    const d: @Vector(3, f32) = switch (face) {
        0 => .{ 1, -t, -s },
        1 => .{ -1, -t, s },
        2 => .{ s, 1, t },
        3 => .{ s, -1, -t },
        4 => .{ s, -t, 1 },
        else => .{ -s, -t, -1 },
    };
    return d / @as(@Vector(3, f32), @splat(@sqrt(@reduce(.Add, d * d))));
}

pub const AtmospherePass = struct {
    context: *const Context,
    allocator: std.mem.Allocator,
    cube: Image,
    // Level zero's six faces as an array, which is what a compute shader
    // writes; the cube's own view is what the scene samples.
    storage_view: vk.ImageView,
    parameters: Buffer,
    module: vk.ShaderModule,
    layout: vk.PipelineLayout,
    pipeline: vk.Pipeline,
    sets: Sets,
    // The environment's sampler, which the cube is bound with.
    sampler: vk.Sampler,
    sun: Sun = .{ .toward_sun = .{ 0, 1, 0 }, .top = 1 },
    // Set by a relight; the next `record` renders the cube and clears it.
    dirty: bool = true,
    // Whether the cube has been rendered once, so its contents and layout are
    // defined. The first render starts from `undefined`.
    initialised: bool = false,

    pub fn init(
        context: *const Context,
        mem: *memory.MemoryAllocator,
        allocator: std.mem.Allocator,
        shader: Shader,
        parameters: Parameters,
        transmittance: Bound,
        multiple: Bound,
        sampler: vk.Sampler,
    ) !AtmospherePass {
        try validate(parameters, transmittance, multiple);

        // Vulkan specification, vkCmdBlitImage: a linear blit needs the
        // format to support blitting both ways and linear filtering.
        const instance = context.instance;
        const lut_features = instance.getPhysicalDeviceFormatProperties(context.physical_device, .r32g32b32a32_sfloat).optimal_tiling_features;
        if (!lut_features.sampled_image_filter_linear_bit) return error.UnsupportedAtmosphereFiltering;
        const cube_features = instance.getPhysicalDeviceFormatProperties(context.physical_device, .r16g16b16a16_sfloat).optimal_tiling_features;
        if (!cube_features.blit_src_bit or !cube_features.blit_dst_bit or !cube_features.sampled_image_filter_linear_bit)
            return error.UnsupportedSkyBlit;

        var cube = try Image.init(context, mem, .{
            .width = extent,
            .height = extent,
            .format = .r16g16b16a16_sfloat,
            .mip_levels = levels,
            .shape = .cube,
            .usage = .{ .storage_bit = true, .sampled_bit = true, .transfer_src_bit = true, .transfer_dst_bit = true },
        });
        errdefer cube.deinit();

        const storage_view = try context.device.createImageView(&.{
            .image = cube.handle,
            .view_type = .@"2d_array",
            .format = cube.format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 6,
            },
        }, null);
        errdefer context.device.destroyImageView(storage_view, null);

        var model = try Buffer.init(context, mem, @sizeOf(Parameters), .{ .storage_buffer_bit = true }, .upload);
        errdefer model.deinit();
        try model.upload(std.mem.asBytes(&parameters));

        var sets = try Sets.init(context, allocator, 1);
        errdefer sets.deinit(context, allocator);

        const module = try pipeline.createModule(context, shader.spirv);
        errdefer context.device.destroyShaderModule(module, null);

        const layout = try pipeline.createLayout(context, .{
            .descriptor_sets = &.{sets.layout},
            .push_constants = &.{push_range},
        });
        errdefer context.device.destroyPipelineLayout(layout, null);

        const built = try pipeline.createCompute(context, .{
            .layout = layout,
            .stage = .{ .module = module, .entry_point = shader.compute_entry },
        });

        sets.writeImages(context, 0, .{
            .transmittance = descriptors.ImageSource{ .view = transmittance.view, .sampler = transmittance.sampler, .layout = .shader_read_only_optimal },
            .multiple = descriptors.ImageSource{ .view = multiple.view, .sampler = multiple.sampler, .layout = .shader_read_only_optimal },
            .cube = descriptors.ImageSource{ .view = storage_view, .sampler = .null_handle, .layout = .general },
        });
        sets.writeBuffers(context, 0, .{ .model = &model });

        return .{
            .context = context,
            .allocator = allocator,
            .cube = cube,
            .storage_view = storage_view,
            .parameters = model,
            .module = module,
            .layout = layout,
            .pipeline = built,
            .sets = sets,
            .sampler = sampler,
        };
    }

    // The caller drains the device before replacing or destroying this pass.
    pub fn deinit(self: *AtmospherePass) void {
        const device = self.context.device;
        device.destroyPipeline(self.pipeline, null);
        device.destroyPipelineLayout(self.layout, null);
        device.destroyShaderModule(self.module, null);
        self.sets.deinit(self.context, self.allocator);
        self.parameters.deinit();
        device.destroyImageView(self.storage_view, null);
        self.cube.deinit();
        self.* = undefined;
    }

    // The cube as the environment's GGX map.
    pub fn bound(self: *const AtmospherePass) Bound {
        return .{ .view = self.cube.view, .sampler = self.sampler, .width = extent, .height = extent, .mip_levels = levels };
    }

    // A new sun for the next frame's render. The direction need not be of unit
    // length; one that is zero or not finite, or a negative irradiance, is
    // refused and leaves the pending sun as it was.
    pub fn relight(self: *AtmospherePass, sun: Sun) error{InvalidSun}!void {
        var scale: f32 = 0;
        for (sun.toward_sun) |c| {
            if (!std.math.isFinite(c)) return error.InvalidSun;
            scale = @max(scale, @abs(c));
        }
        if (scale == 0 or !std.math.isFinite(sun.top) or sun.top < 0) return error.InvalidSun;
        // Scaled by the largest component first, so squaring cannot overflow.
        const d = @as(@Vector(3, f32), sun.toward_sun) / @as(@Vector(3, f32), @splat(scale));
        const toward = d / @as(@Vector(3, f32), @splat(@sqrt(@reduce(.Add, d * d))));
        self.sun = .{ .toward_sun = toward, .top = sun.top };
        self.dirty = true;
    }

    // Renders the cube and its mips if the sun moved since the last render;
    // records nothing otherwise. In the frame's own command buffer, before the
    // main pass that samples it.
    //
    // One cube shared by every frame in flight, with no copy per frame: Vulkan
    // specification, Synchronization and Cache Control, Pipeline Barriers
    // (vkCmdPipelineBarrier2) and Submission Order. A barrier's first scope
    // covers commands submitted earlier on the same queue, so the barrier from
    // the fragment stage below orders this frame's writes after the previous
    // frame's reads of the cube.
    pub fn record(self: *AtmospherePass, cmd: vk.CommandBuffer) void {
        if (!self.dirty) return;
        const device = self.context.device;

        // The tables' upload is a copy; the march samples them.
        commands.recordMemoryBarrier(self.context, cmd, .{
            .src_stage = .{ .copy_bit = true },
            .src_access = .{ .transfer_write_bit = true },
            .dst_stage = .{ .compute_shader_bit = true },
            .dst_access = .{ .shader_sampled_read_bit = true },
        });

        const before: vk.ImageLayout = if (self.initialised) .shader_read_only_optimal else .undefined;
        // Level zero to be written by the march, the rest by the blits. The
        // previous contents are discarded either way: nothing reads them.
        self.barrier(
            cmd,
            0,
            1,
            before,
            .general,
            .{ .fragment_shader_bit = true },
            .{},
            .{ .compute_shader_bit = true },
            .{ .shader_storage_write_bit = true },
        );
        self.barrier(
            cmd,
            1,
            levels - 1,
            before,
            .transfer_dst_optimal,
            .{ .fragment_shader_bit = true },
            .{},
            .{ .blit_bit = true },
            .{ .transfer_write_bit = true },
        );

        device.cmdBindPipeline(cmd, .compute, self.pipeline);
        device.cmdBindDescriptorSets(cmd, .compute, self.layout, 0, &.{self.sets.set(0)}, &.{});
        device.cmdPushConstants(cmd, self.layout, push_range.stage_flags, 0, @sizeOf(Sun), @ptrCast(&self.sun));
        // Eight by eight texels a group, the shader's local size; a face a
        // layer of the dispatch.
        device.cmdDispatch(cmd, extent / 8, extent / 8, 6);

        self.barrier(
            cmd,
            0,
            1,
            .general,
            .transfer_src_optimal,
            .{ .compute_shader_bit = true },
            .{ .shader_storage_write_bit = true },
            .{ .blit_bit = true },
            .{ .transfer_read_bit = true },
        );
        // Each level from the one above it, which then becomes the source of
        // the next.
        for (1..levels) |level| {
            const mip: u32 = @intCast(level);
            const source_side: i32 = @intCast(extent >> @as(u5, @intCast(level - 1)));
            const side = @divExact(source_side, 2);
            const regions = [_]vk.ImageBlit{.{
                .src_subresource = .{ .aspect_mask = .{ .color_bit = true }, .mip_level = mip - 1, .base_array_layer = 0, .layer_count = 6 },
                .src_offsets = .{ .{ .x = 0, .y = 0, .z = 0 }, .{ .x = source_side, .y = source_side, .z = 1 } },
                .dst_subresource = .{ .aspect_mask = .{ .color_bit = true }, .mip_level = mip, .base_array_layer = 0, .layer_count = 6 },
                .dst_offsets = .{ .{ .x = 0, .y = 0, .z = 0 }, .{ .x = side, .y = side, .z = 1 } },
            }};
            device.cmdBlitImage(cmd, self.cube.handle, .transfer_src_optimal, self.cube.handle, .transfer_dst_optimal, &regions, .linear);
            self.barrier(
                cmd,
                mip,
                1,
                .transfer_dst_optimal,
                .transfer_src_optimal,
                .{ .blit_bit = true },
                .{ .transfer_write_bit = true },
                .{ .blit_bit = true },
                .{ .transfer_read_bit = true },
            );
        }

        // Every level is now a blit source; the scene samples them all.
        self.barrier(
            cmd,
            0,
            levels,
            .transfer_src_optimal,
            .shader_read_only_optimal,
            .{ .compute_shader_bit = true, .blit_bit = true },
            .{ .shader_storage_write_bit = true, .transfer_write_bit = true },
            .{ .fragment_shader_bit = true },
            .{ .shader_sampled_read_bit = true },
        );
        self.initialised = true;
        self.dirty = false;
    }

    // Copies level zero into `destination`, rendering first if the sun has
    // moved. A cold diagnostic: the caller submits, waits, and only then maps
    // the buffer, which receives the six faces as tightly packed RGBA16F.
    pub fn recordReadback(self: *AtmospherePass, cmd: vk.CommandBuffer, destination: *Buffer) !void {
        if (destination.context.device.handle != self.context.device.handle or
            destination.size < extent * extent * 6 * 8 or !destination.usage.transfer_dst_bit) return error.InvalidSkyReadback;
        self.record(cmd);
        self.barrier(
            cmd,
            0,
            1,
            .shader_read_only_optimal,
            .transfer_src_optimal,
            .{ .all_commands_bit = true },
            .{ .memory_write_bit = true, .memory_read_bit = true },
            .{ .copy_bit = true },
            .{ .transfer_read_bit = true },
        );
        const regions = [_]vk.BufferImageCopy{.{
            .buffer_offset = 0,
            .buffer_row_length = 0,
            .buffer_image_height = 0,
            .image_subresource = .{ .aspect_mask = .{ .color_bit = true }, .mip_level = 0, .base_array_layer = 0, .layer_count = 6 },
            .image_offset = .{ .x = 0, .y = 0, .z = 0 },
            .image_extent = .{ .width = extent, .height = extent, .depth = 1 },
        }};
        self.context.device.cmdCopyImageToBuffer(cmd, self.cube.handle, .transfer_src_optimal, destination.handle, &regions);
        self.barrier(
            cmd,
            0,
            1,
            .transfer_src_optimal,
            .shader_read_only_optimal,
            .{ .copy_bit = true },
            .{},
            .{ .fragment_shader_bit = true },
            .{ .shader_sampled_read_bit = true },
        );
        commands.recordMemoryBarrier(self.context, cmd, .{
            .src_stage = .{ .copy_bit = true },
            .src_access = .{ .transfer_write_bit = true },
            .dst_stage = .{ .host_bit = true },
            .dst_access = .{ .host_read_bit = true },
        });
    }

    // One layout transition over `count` levels from `mip`, all six faces.
    fn barrier(
        self: *const AtmospherePass,
        cmd: vk.CommandBuffer,
        mip: u32,
        count: u32,
        old: vk.ImageLayout,
        new: vk.ImageLayout,
        src_stage: vk.PipelineStageFlags2,
        src_access: vk.AccessFlags2,
        dst_stage: vk.PipelineStageFlags2,
        dst_access: vk.AccessFlags2,
    ) void {
        const barriers = [_]vk.ImageMemoryBarrier2{.{
            .src_stage_mask = src_stage,
            .src_access_mask = src_access,
            .dst_stage_mask = dst_stage,
            .dst_access_mask = dst_access,
            .old_layout = old,
            .new_layout = new,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = self.cube.handle,
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = mip,
                .level_count = count,
                .base_array_layer = 0,
                .layer_count = 6,
            },
        }};
        self.context.device.cmdPipelineBarrier2(cmd, &.{ .image_memory_barrier_count = 1, .p_image_memory_barriers = &barriers });
    }
};

// What the host handed over is checked once, here, before anything is made
// from it. The march divides by the radii and the scale heights and raises the
// phase to a power of 1 - g, and the tables' sizes are what the shader's uv
// mappings assume.
fn validate(parameters: Parameters, transmittance: Bound, multiple: Bound) !void {
    for (parameters.rows) |row| for (row) |value| {
        if (!std.math.isFinite(value)) return error.InvalidAtmosphereParameters;
    };
    const radii = parameters.rows[0];
    const samples = parameters.rows[7];
    if (radii[0] <= 0 or radii[1] <= radii[0] or radii[2] <= 0 or radii[3] <= 0 or
        @abs(parameters.rows[6][3]) >= 1 or
        samples[0] < 1 or samples[1] < samples[0] or samples[1] > 1024 or
        samples[2] < 0 or samples[2] > radii[1] - radii[0]) return error.InvalidAtmosphereParameters;
    if (transmittance.width != 256 or transmittance.height != 64 or transmittance.mip_levels != 1 or
        multiple.width != 32 or multiple.height != 32 or multiple.mip_levels != 1) return error.InvalidAtmosphereTables;
}
