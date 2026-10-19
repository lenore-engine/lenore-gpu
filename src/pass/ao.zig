const std = @import("std");
const vk = @import("vulkan");

const Context = @import("../device/context.zig").Context;
const descriptors = @import("../binding/descriptors.zig");
const image = @import("../object/image.zig");
const memory = @import("../memory/allocator.zig");
const pass = @import("scene.zig");
const pipeline = @import("../binding/pipeline.zig");

const Allocator = std.mem.Allocator;

// Screen-space ambient occlusion, between the depth prepass and the shaded
// pass: an estimate of how much ambient each pixel receives, from the depth
// the prepass stored, at half the target's resolution and blurred along each
// axis with the depth as the guide. The main pass reads the result through the
// scene set and scales its ambient and environment term by it.
//
// This owns the half-resolution images, the samplers, the descriptor sets and
// the four pipelines. The camera depth is the renderer's; `recordEstimate`
// moves it into a read-only layout for the one draw that reads it and back
// into the attachment layout the shaded pass loads it in.
//
// First the distance chain: level zero from the camera depth, then each level
// from the one above it, one draw a level. Then three draws: the estimate into
// `estimate`, reading the chain, the X blur from there into `crossed`, and the
// Y blur from `crossed` back into `estimate`, which is then what the main pass
// samples. No draw reads the image or the level it writes.

// The format of both images. One channel of eight bits is enough for a factor
// in [0, 1] that is blurred and then multiplied into an ambient term, and
// R8_UNORM is among the formats every implementation must render to and
// sample (Vulkan specification, Required Format Support, the mandatory
// colour attachment and sampled image features).
pub const format: vk.Format = .r8_unorm;

const usage: vk.ImageUsageFlags = .{ .color_attachment_bit = true, .sampled_bit = true };

// The distance chain's format. The distances are compared against each other
// and against a radius, so they keep single precision. Vulkan specification,
// Required Format Support, table "Mandatory Format Support: 32-bit
// Components": R32_SFLOAT must be sampled and rendered to; it need not be
// filtered linearly, which nothing here asks of it.
pub const distance_format: vk.Format = .r32_sfloat;

// Below this many texels on a side a level is not built: a tap that far out
// is past any disk at half resolution, and each level costs a draw.
pub const min_level_extent: u32 = 8;

// The chain's ceiling, which bounds the descriptor sets allocated once. Ten
// halvings from 8 texels is 8192, past any half-resolution target.
pub const max_levels: u32 = 11;

// How many levels the chain for a half-resolution extent has: halving until a
// side would drop under `min_level_extent`, and at least level zero.
pub fn levelCount(half: vk.Extent2D) u32 {
    var count: u32 = 1;
    while (count < max_levels) : (count += 1) {
        if (image.mipExtent(half.width, count) < min_level_extent or
            image.mipExtent(half.height, count) < min_level_extent) break;
    }
    return count;
}

// Mirrors `AoPushConstants` in ao.slang. Both fragment entry points read it
// whole, against one layout.
pub const PushConstants = extern struct {
    texel: [2]f32,
    half_width: f32,
    half_height: f32,
    depth_scale: f32,
    depth_offset: f32,
    radius: f32,
    pixels_per_unit: f32,
    intensity: f32,
    bias: f32,
    direction: [2]f32,
    // `Camera.jitter`'s offset, which the reconstruction takes back out.
    jitter: [2]f32,
};

pub const push_constant_range: vk.PushConstantRange = .{
    .stage_flags = .{ .fragment_bit = true },
    .offset = 0,
    .size = @sizeOf(PushConstants),
};

// What the frame decides, the rest of the push block being the pass's own.
pub const Frame = struct {
    // `AmbientOcclusionReconstruction` in lenore-scene, for the frame's
    // projection. `pixels_per_unit` is at the full target's height; the pass
    // halves it for the half-resolution estimate.
    half_width: f32,
    half_height: f32,
    depth_scale: f32,
    depth_offset: f32,
    pixels_per_unit: f32,
    radius: f32,
    intensity: f32,
    bias: f32,
    // The sub-pixel offset the frame's projection carries, in the framebuffer's
    // device coordinates; zero when unjittered.
    jitter: [2]f32 = .{ 0, 0 },
};

pub const bindings = [_]descriptors.Binding{
    .{ .slot = 0, .name = "depth_map", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
    .{ .slot = 1, .name = "source", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
    .{ .slot = 2, .name = "distances", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
};

pub const Sets = descriptors.Sets(&bindings);

// One set per draw, each naming the images that draw reads. Every binding
// still has to hold a valid descriptor where a draw does not read it: the
// estimate's source names `crossed`, and the chain's sets name `crossed` and,
// at level zero, the whole chain, none of which those draws sample.
const estimate_set: u32 = 0;
const cross_set: u32 = 1;
const back_set: u32 = 2;
// The set the draw writing chain level `level` binds: level zero reads the
// depth, every other the level above it.
fn levelSet(level: u32) u32 {
    return 3 + level;
}
const set_count: u32 = 3 + max_levels;

// The result's slot in the scene set, which the renderer joins with the
// others there; a slot claimed twice is a compile error in that list.
pub const scene_bindings = [_]descriptors.Binding{
    .{ .slot = 6, .name = "ambient_occlusion", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
};

// Points the scene set's slot at the result. At init and at each resize, when
// no frame is in flight.
pub fn writeResult(context: *const Context, scene_set: vk.DescriptorSet, view: vk.ImageView, sampler: vk.Sampler) void {
    const info = vk.DescriptorImageInfo{ .sampler = sampler, .image_view = view, .image_layout = pass.sampled_layout };
    const writes = [_]vk.WriteDescriptorSet{.{
        .dst_set = scene_set,
        .dst_binding = scene_bindings[0].slot,
        .dst_array_element = 0,
        .descriptor_count = 1,
        .descriptor_type = scene_bindings[0].kind,
        .p_image_info = @ptrCast(&info),
        .p_buffer_info = &no_buffers,
        .p_texel_buffer_view = &no_texel_buffers,
    }};
    context.device.updateDescriptorSets(&writes, null);
}

pub const vertex_count: u32 = 3;

// The layout the camera depth is read in. Read-only rather than the shader
// layout the post pass uses: the shaded pass loads it again as an attachment
// straight after, and this is the layout that serves both a depth test and a
// sample without a write.
pub const depth_read_layout: vk.ImageLayout = .depth_read_only_optimal;

pub const InitError = image.InitError ||
    image.MipViewError ||
    descriptors.InitError ||
    pipeline.CreateError ||
    vk.DeviceWrapper.CreateSamplerError;

pub const Shader = struct {
    spirv: []const u32,
    vertex_entry: [*:0]const u8,
    distance_entry: [*:0]const u8,
    downsample_entry: [*:0]const u8,
    occlusion_entry: [*:0]const u8,
    blur_entry: [*:0]const u8,
};

// Half of each side, rounding up so an odd target keeps its last column.
pub fn halfExtent(target: vk.Extent2D) vk.Extent2D {
    return .{
        .width = @max(1, (target.width + 1) / 2),
        .height = @max(1, (target.height + 1) / 2),
    };
}

// The push block for one draw. The estimate's step is zero; the blur's is one
// texel along an axis. `pixels_per_unit` follows the half-resolution extent,
// since the sample offsets are measured in the texels being written.
pub fn pushFor(frame: Frame, extent: vk.Extent2D, direction: [2]f32) PushConstants {
    const width: f32 = @floatFromInt(extent.width);
    const height: f32 = @floatFromInt(extent.height);
    return .{
        .texel = .{ 1 / width, 1 / height },
        .half_width = frame.half_width,
        .half_height = frame.half_height,
        .depth_scale = frame.depth_scale,
        .depth_offset = frame.depth_offset,
        .radius = frame.radius,
        .pixels_per_unit = frame.pixels_per_unit * 0.5,
        .intensity = frame.intensity,
        .bias = frame.bias,
        .direction = direction,
        .jitter = frame.jitter,
    };
}

pub const AoPass = struct {
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    allocator: Allocator,

    // Persistent across a resize.
    depth_sampler: vk.Sampler,
    linear_sampler: vk.Sampler,
    module: vk.ShaderModule,
    layout: vk.PipelineLayout,
    distance_pipeline: vk.Pipeline,
    downsample_pipeline: vk.Pipeline,
    occlusion_pipeline: vk.Pipeline,
    blur_pipeline: vk.Pipeline,
    sets: Sets,

    // Replaced by a resize. The chain is one image of `level_count` levels and
    // a view of each, since a level is written and read one at a time; its
    // own view spans every level and is what the estimate and the blur read.
    estimate: image.Image,
    crossed: image.Image,
    chain: image.Image,
    level_views: [max_levels]vk.ImageView,
    level_count: u32,

    pub fn init(
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        allocator: Allocator,
        target: vk.Extent2D,
        depth_view: vk.ImageView,
        shader: Shader,
    ) InitError!AoPass {
        const depth_sampler = try createSampler(context, .nearest);
        errdefer context.device.destroySampler(depth_sampler, null);
        const linear_sampler = try createSampler(context, .linear);
        errdefer context.device.destroySampler(linear_sampler, null);

        const module = try pipeline.createModule(context, shader.spirv);
        errdefer context.device.destroyShaderModule(module, null);

        var sets = try Sets.init(context, allocator, set_count);
        errdefer sets.deinit(context, allocator);

        const layout = try pipeline.createLayout(context, .{
            .descriptor_sets = &.{sets.layout},
            .push_constants = &.{push_constant_range},
        });
        errdefer context.device.destroyPipelineLayout(layout, null);

        const distance_pipeline = try pipeline.create(context, config(layout, module, shader.vertex_entry, shader.distance_entry, distance_format));
        errdefer context.device.destroyPipeline(distance_pipeline, null);
        const downsample_pipeline = try pipeline.create(context, config(layout, module, shader.vertex_entry, shader.downsample_entry, distance_format));
        errdefer context.device.destroyPipeline(downsample_pipeline, null);
        const occlusion_pipeline = try pipeline.create(context, config(layout, module, shader.vertex_entry, shader.occlusion_entry, format));
        errdefer context.device.destroyPipeline(occlusion_pipeline, null);
        const blur_pipeline = try pipeline.create(context, config(layout, module, shader.vertex_entry, shader.blur_entry, format));
        errdefer context.device.destroyPipeline(blur_pipeline, null);

        var self: AoPass = .{
            .context = context,
            .memory_allocator = memory_allocator,
            .allocator = allocator,
            .depth_sampler = depth_sampler,
            .linear_sampler = linear_sampler,
            .module = module,
            .layout = layout,
            .distance_pipeline = distance_pipeline,
            .downsample_pipeline = downsample_pipeline,
            .occlusion_pipeline = occlusion_pipeline,
            .blur_pipeline = blur_pipeline,
            .sets = sets,
            .estimate = undefined,
            .crossed = undefined,
            .chain = undefined,
            .level_views = undefined,
            .level_count = 0,
        };
        try self.createImages(target, depth_view);
        return self;
    }

    // Vulkan specification, vkDestroyPipeline and the rest: every submission
    // naming any of these must have completed. The caller drains the device.
    pub fn deinit(self: *AoPass) void {
        const device = self.context.device;
        self.destroyImages();
        device.destroyPipeline(self.blur_pipeline, null);
        device.destroyPipeline(self.occlusion_pipeline, null);
        device.destroyPipeline(self.downsample_pipeline, null);
        device.destroyPipeline(self.distance_pipeline, null);
        device.destroyPipelineLayout(self.layout, null);
        self.sets.deinit(self.context, self.allocator);
        device.destroyShaderModule(self.module, null);
        device.destroySampler(self.linear_sampler, null);
        device.destroySampler(self.depth_sampler, null);
        self.* = undefined;
    }

    // Rebuild the images for a new target and a new depth view. Candidate
    // first, as `BloomPass.recreate` is: a resize that cannot be honoured
    // leaves a pass that still draws.
    pub fn recreate(self: *AoPass, target: vk.Extent2D, depth_view: vk.ImageView) InitError!void {
        const previous = self.*;
        self.createImages(target, depth_view) catch |err| {
            self.* = previous;
            return err;
        };
        var old = previous;
        old.destroyImages();
    }

    fn createImages(self: *AoPass, target: vk.Extent2D, depth_view: vk.ImageView) InitError!void {
        const half = halfExtent(target);
        var estimate = try image.Image.init(self.context, self.memory_allocator, .{
            .width = half.width,
            .height = half.height,
            .format = format,
            .usage = usage,
        });
        errdefer estimate.deinit();
        var crossed = try image.Image.init(self.context, self.memory_allocator, .{
            .width = half.width,
            .height = half.height,
            .format = format,
            .usage = usage,
        });
        errdefer crossed.deinit();

        const levels = levelCount(half);
        var chain = try image.Image.init(self.context, self.memory_allocator, .{
            .width = half.width,
            .height = half.height,
            .format = distance_format,
            .mip_levels = levels,
            .usage = usage,
        });
        errdefer chain.deinit();
        var views: [max_levels]vk.ImageView = undefined;
        var created: u32 = 0;
        errdefer for (views[0..created]) |view| self.context.device.destroyImageView(view, null);
        while (created < levels) : (created += 1) views[created] = try chain.createMipView(created);

        self.estimate = estimate;
        self.crossed = crossed;
        self.chain = chain;
        self.level_views = views;
        self.level_count = levels;
        self.writeSets(depth_view);
    }

    fn destroyImages(self: *AoPass) void {
        for (self.level_views[0..self.level_count]) |view| self.context.device.destroyImageView(view, null);
        self.chain.deinit();
        self.crossed.deinit();
        self.estimate.deinit();
    }

    // Safe only when no frame is in flight, which is where images are built.
    // The chain's sets past its depth are left as they were allocated: nothing
    // binds them, because `recordEstimate` walks the levels this chain has.
    fn writeSets(self: *const AoPass, depth_view: vk.ImageView) void {
        const written = 3 + self.level_count;
        var infos: [set_count * bindings.len]vk.DescriptorImageInfo = undefined;
        var writes: [set_count * bindings.len]vk.WriteDescriptorSet = undefined;
        for (0..written) |set| {
            const level = if (set >= levelSet(0)) set - levelSet(0) else null;
            const source = if (set == cross_set) self.estimate.view else self.crossed.view;
            const chain_view = if (level) |l| (if (l == 0) self.chain.view else self.level_views[l - 1]) else self.chain.view;
            const views = [bindings.len]vk.DescriptorImageInfo{
                .{ .sampler = self.depth_sampler, .image_view = depth_view, .image_layout = depth_read_layout },
                .{ .sampler = self.depth_sampler, .image_view = source, .image_layout = pass.sampled_layout },
                .{ .sampler = self.depth_sampler, .image_view = chain_view, .image_layout = pass.sampled_layout },
            };
            for (bindings, views, 0..) |binding, info, index| {
                const at = set * bindings.len + index;
                infos[at] = info;
                writes[at] = .{
                    .dst_set = self.sets.set(@intCast(set)),
                    .dst_binding = binding.slot,
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = binding.kind,
                    .p_image_info = @ptrCast(&infos[at]),
                    .p_buffer_info = &no_buffers,
                    .p_texel_buffer_view = &no_texel_buffers,
                };
            }
        }
        self.context.device.updateDescriptorSets(writes[0 .. written * bindings.len], null);
    }

    // What the main pass samples, and how: bilinear, since it reads a
    // half-resolution image at full resolution.
    pub fn resultView(self: *const AoPass) vk.ImageView {
        return self.estimate.view;
    }

    pub fn resultSampler(self: *const AoPass) vk.Sampler {
        return self.linear_sampler;
    }

    fn extent(self: *const AoPass) vk.Extent2D {
        return .{ .width = self.estimate.width, .height = self.estimate.height };
    }

    // The distance chain and the estimate, between the prepass and the shaded
    // pass. On entry the camera depth is the prepass's attachment, and on exit
    // it is that attachment again: level zero is the one draw that reads it.
    // On exit every level of the chain is sampled and `estimate` holds the raw
    // estimate as an attachment, for `recordBlur`.
    //
    // Two calls rather than one so that the frame can time the estimate and
    // the blur apart.
    pub fn recordEstimate(self: *const AoPass, command_buffer: vk.CommandBuffer, depth_image: vk.Image, frame: Frame) void {
        const device = self.context.device;
        const half = self.extent();
        const push = pushFor(frame, half, .{ 0, 0 });

        const opening = [_]vk.ImageMemoryBarrier2{
            depthToRead(depth_image),
            levelBarrier(self.chain.handle, 0, .discard_to_attachment),
        };
        device.cmdPipelineBarrier2(command_buffer, &.{
            .image_memory_barrier_count = opening.len,
            .p_image_memory_barriers = &opening,
        });
        self.draw(command_buffer, self.distance_pipeline, levelSet(0), self.level_views[0], half, push);

        var level: u32 = 1;
        while (level < self.level_count) : (level += 1) {
            var stepping = [_]vk.ImageMemoryBarrier2{
                levelBarrier(self.chain.handle, level - 1, .attachment_to_sampled),
                levelBarrier(self.chain.handle, level, .discard_to_attachment),
                undefined,
            };
            var count: u32 = 2;
            // The depth's last reader was the draw just before.
            if (level == 1) {
                stepping[2] = depthToAttachment(depth_image);
                count = 3;
            }
            device.cmdPipelineBarrier2(command_buffer, &.{
                .image_memory_barrier_count = count,
                .p_image_memory_barriers = &stepping,
            });
            self.draw(command_buffer, self.downsample_pipeline, levelSet(level), self.level_views[level], self.chain.levelExtent(level), push);
        }

        var closing = [_]vk.ImageMemoryBarrier2{
            levelBarrier(self.chain.handle, self.level_count - 1, .attachment_to_sampled),
            colourBarrier(self.estimate.handle, .discard_to_attachment),
            undefined,
        };
        var count: u32 = 2;
        if (self.level_count == 1) {
            closing[2] = depthToAttachment(depth_image);
            count = 3;
        }
        device.cmdPipelineBarrier2(command_buffer, &.{
            .image_memory_barrier_count = count,
            .p_image_memory_barriers = &closing,
        });
        self.draw(command_buffer, self.occlusion_pipeline, estimate_set, self.estimate.view, half, push);
    }

    // The two blurs, after `recordEstimate` and with the same frame. On exit
    // `estimate` is in the sampled layout for the main pass.
    pub fn recordBlur(self: *const AoPass, command_buffer: vk.CommandBuffer, frame: Frame) void {
        const device = self.context.device;
        const half = self.extent();

        const crossing = [_]vk.ImageMemoryBarrier2{
            colourBarrier(self.estimate.handle, .attachment_to_sampled),
            colourBarrier(self.crossed.handle, .discard_to_attachment),
        };
        device.cmdPipelineBarrier2(command_buffer, &.{
            .image_memory_barrier_count = crossing.len,
            .p_image_memory_barriers = &crossing,
        });
        self.draw(command_buffer, self.blur_pipeline, cross_set, self.crossed.view, half, pushFor(frame, half, .{ 1, 0 }));

        const back = [_]vk.ImageMemoryBarrier2{
            colourBarrier(self.crossed.handle, .attachment_to_sampled),
            colourBarrier(self.estimate.handle, .sampled_to_attachment),
        };
        device.cmdPipelineBarrier2(command_buffer, &.{
            .image_memory_barrier_count = back.len,
            .p_image_memory_barriers = &back,
        });
        self.draw(command_buffer, self.blur_pipeline, back_set, self.estimate.view, half, pushFor(frame, half, .{ 0, 1 }));

        const closing = colourBarrier(self.estimate.handle, .attachment_to_sampled);
        device.cmdPipelineBarrier2(command_buffer, &.{
            .image_memory_barrier_count = 1,
            .p_image_memory_barriers = @ptrCast(&closing),
        });
    }

    // With the pass off, the result still has to read as unoccluded and sit in
    // the layout the main pass's descriptor names. Cleared to one by a
    // rendering with no draw, which is the same three barriers' worth of work
    // as a transfer clear and stays within the usages the image has.
    pub fn recordClear(self: *const AoPass, command_buffer: vk.CommandBuffer) void {
        const device = self.context.device;
        const opening = colourBarrier(self.estimate.handle, .discard_to_attachment);
        device.cmdPipelineBarrier2(command_buffer, &.{
            .image_memory_barrier_count = 1,
            .p_image_memory_barriers = @ptrCast(&opening),
        });
        const colour = vk.RenderingAttachmentInfo{
            .image_view = self.estimate.view,
            .image_layout = .color_attachment_optimal,
            .resolve_mode = .{},
            .resolve_image_layout = .undefined,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = .{ .color = .{ .float_32 = .{ 1, 1, 1, 1 } } },
        };
        device.cmdBeginRendering(command_buffer, &.{
            .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = self.extent() },
            .layer_count = 1,
            .view_mask = 0,
            .color_attachment_count = 1,
            .p_color_attachments = @ptrCast(&colour),
        });
        device.cmdEndRendering(command_buffer);
        const closing = colourBarrier(self.estimate.handle, .attachment_to_sampled);
        device.cmdPipelineBarrier2(command_buffer, &.{
            .image_memory_barrier_count = 1,
            .p_image_memory_barriers = @ptrCast(&closing),
        });
    }

    fn draw(
        self: *const AoPass,
        command_buffer: vk.CommandBuffer,
        selected: vk.Pipeline,
        set: u32,
        target: vk.ImageView,
        area: vk.Extent2D,
        push: PushConstants,
    ) void {
        const device = self.context.device;
        const colour = vk.RenderingAttachmentInfo{
            .image_view = target,
            .image_layout = .color_attachment_optimal,
            .resolve_mode = .{},
            .resolve_image_layout = .undefined,
            // Every texel is written by the covering triangle.
            .load_op = .dont_care,
            .store_op = .store,
            .clear_value = undefined,
        };
        device.cmdBeginRendering(command_buffer, &.{
            .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = area },
            .layer_count = 1,
            .view_mask = 0,
            .color_attachment_count = 1,
            .p_color_attachments = @ptrCast(&colour),
        });
        device.cmdSetViewport(command_buffer, 0, &.{pass.viewport(area)});
        device.cmdSetScissor(command_buffer, 0, &.{pass.scissor(area)});
        device.cmdBindPipeline(command_buffer, .graphics, selected);
        device.cmdBindDescriptorSets(command_buffer, .graphics, self.layout, 0, &.{self.sets.set(set)}, &.{});
        var block = push;
        device.cmdPushConstants(
            command_buffer,
            self.layout,
            push_constant_range.stage_flags,
            push_constant_range.offset,
            push_constant_range.size,
            @ptrCast(&block),
        );
        device.cmdDraw(command_buffer, vertex_count, 1, 0, 0);
        device.cmdEndRendering(command_buffer);
    }
};

fn config(layout: vk.PipelineLayout, module: vk.ShaderModule, vertex_entry: [*:0]const u8, fragment_entry: [*:0]const u8, colour: vk.Format) pipeline.Config {
    return .{
        .mode = .solid,
        .vertex_input = .none,
        .culling = .{ .fixed = .{} },
        .formats = .{ .colour = colour },
        .layout = layout,
        .stages = .{
            .vertex = .{ .module = module, .entry_point = vertex_entry },
            .fragment = .{ .module = module, .entry_point = fragment_entry },
        },
    };
}

const depth_range = vk.ImageSubresourceRange{
    .aspect_mask = .{ .depth_bit = true },
    .base_mip_level = 0,
    .level_count = 1,
    .base_array_layer = 0,
    .layer_count = 1,
};

const level_range_template = vk.ImageSubresourceRange{
    .aspect_mask = .{ .color_bit = true },
    .base_mip_level = 0,
    .level_count = 1,
    .base_array_layer = 0,
    .layer_count = 1,
};

// The prepass's depth writes made visible to the distance draw's sampling, in
// the read-only layout.
pub fn depthToRead(handle: vk.Image) vk.ImageMemoryBarrier2 {
    return .{
        .src_stage_mask = .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true },
        .src_access_mask = .{ .depth_stencil_attachment_write_bit = true },
        .dst_stage_mask = .{ .fragment_shader_bit = true },
        .dst_access_mask = .{ .shader_sampled_read_bit = true },
        .old_layout = .depth_attachment_optimal,
        .new_layout = depth_read_layout,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = handle,
        .subresource_range = depth_range,
    };
}

// Back to the attachment the shaded pass loads and may add to, after the
// sampling: a write-after-read, and the layout change the load needs.
pub fn depthToAttachment(handle: vk.Image) vk.ImageMemoryBarrier2 {
    return .{
        .src_stage_mask = .{ .fragment_shader_bit = true },
        .src_access_mask = .{ .shader_sampled_read_bit = true },
        .dst_stage_mask = .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true },
        .dst_access_mask = .{
            .depth_stencil_attachment_read_bit = true,
            .depth_stencil_attachment_write_bit = true,
        },
        .old_layout = depth_read_layout,
        .new_layout = .depth_attachment_optimal,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = handle,
        .subresource_range = depth_range,
    };
}

pub const ColourTransition = enum {
    // The first write this frame. Undefined as the old layout discards what the
    // draw is about to replace; the source scope is the previous frame's read
    // by the main pass, the estimate, the blur or the level below.
    discard_to_attachment,
    attachment_to_sampled,
    sampled_to_attachment,
};

pub fn colourBarrier(handle: vk.Image, transition: ColourTransition) vk.ImageMemoryBarrier2 {
    return levelBarrier(handle, 0, transition);
}

// One level of an image, as `colourBarrier` moves a whole single-level one.
// The chain's levels are written and read one at a time, so each moves alone.
pub fn levelBarrier(handle: vk.Image, level: u32, transition: ColourTransition) vk.ImageMemoryBarrier2 {
    var range = level_range_template;
    range.base_mip_level = level;
    var barrier: vk.ImageMemoryBarrier2 = .{
        .src_stage_mask = .{},
        .src_access_mask = .{},
        .dst_stage_mask = .{},
        .dst_access_mask = .{},
        .old_layout = .undefined,
        .new_layout = .undefined,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = handle,
        .subresource_range = range,
    };
    switch (transition) {
        .discard_to_attachment => {
            barrier.src_stage_mask = .{ .fragment_shader_bit = true };
            barrier.src_access_mask = .{ .shader_sampled_read_bit = true };
            barrier.dst_stage_mask = .{ .color_attachment_output_bit = true };
            barrier.dst_access_mask = .{ .color_attachment_write_bit = true };
            barrier.old_layout = .undefined;
            barrier.new_layout = .color_attachment_optimal;
        },
        .attachment_to_sampled => {
            barrier.src_stage_mask = .{ .color_attachment_output_bit = true };
            barrier.src_access_mask = .{ .color_attachment_write_bit = true };
            barrier.dst_stage_mask = .{ .fragment_shader_bit = true };
            barrier.dst_access_mask = .{ .shader_sampled_read_bit = true };
            barrier.old_layout = .color_attachment_optimal;
            barrier.new_layout = pass.sampled_layout;
        },
        .sampled_to_attachment => {
            barrier.src_stage_mask = .{ .fragment_shader_bit = true };
            barrier.src_access_mask = .{ .shader_sampled_read_bit = true };
            barrier.dst_stage_mask = .{ .color_attachment_output_bit = true };
            barrier.dst_access_mask = .{ .color_attachment_write_bit = true };
            barrier.old_layout = pass.sampled_layout;
            barrier.new_layout = .color_attachment_optimal;
        },
    }
    return barrier;
}

// Clamped to the edge, so a sample past the border repeats it rather than
// reading a border colour as a depth or an estimate.
fn createSampler(context: *const Context, filter: vk.Filter) vk.DeviceWrapper.CreateSamplerError!vk.Sampler {
    return context.device.createSampler(&.{
        .mag_filter = filter,
        .min_filter = filter,
        .mipmap_mode = .nearest,
        .address_mode_u = .clamp_to_edge,
        .address_mode_v = .clamp_to_edge,
        .address_mode_w = .clamp_to_edge,
        .mip_lod_bias = 0,
        .anisotropy_enable = .false,
        .max_anisotropy = 1,
        .compare_enable = .false,
        .compare_op = .always,
        .min_lod = 0,
        .max_lod = 0,
        .border_color = .float_opaque_white,
        .unnormalized_coordinates = .false,
    }, null);
}

// Vulkan specification, VkWriteDescriptorSet: the members not selected by
// descriptorType are ignored, but the pointers are not optional.
const no_buffers = [_]vk.DescriptorBufferInfo{};
const no_texel_buffers = [_]vk.BufferView{};
