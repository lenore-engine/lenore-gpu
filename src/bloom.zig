const std = @import("std");
const vk = @import("vulkan");

const Context = @import("context.zig").Context;
const descriptors = @import("descriptors.zig");
const image = @import("image.zig");
const memory = @import("memory/allocator.zig");
const pass = @import("pass.zig");
const pipeline = @import("pipeline.zig");

const Allocator = std.mem.Allocator;

// Bloom: the light a bright surface spills into the pixels around it.
//
// The HDR target is reduced through a chain of levels, each half the one before,
// and then walked back up with each level added onto the one above it. What that
// buys over a single blur is the shape of the falloff. A lens does not produce a
// Gaussian halo, it produces one with a long tail, and a sum of scales with a
// decaying weight is a tail; one blur of one radius is a blob whose size does
// not change with how bright the source is.
//
// This pass owns the chain, its sampler and its two pipelines. It does not own
// the composite: level zero is added into the scene by the post pass, before the
// tone map operator, so the glow is compressed by the same curve as the surface
// under it. A composite of its own would be a third full-resolution pass to
// write what the post pass is already writing.

// Where the chain stops.
//
// A level's halo covers twice the screen fraction of the level above it, so the
// deepest ones are spread over the whole picture and read as a lift of the black
// level rather than as glow around anything, while each still costs a draw and
// two barriers. Eight is where that is already true.
//
// The number is the reference's rather than a measurement of ours. What would
// replace it is a picture at two floors with the difference looked for, and the
// difference is expected to be invisible, which is why it has not been made a
// decision to close.
const min_level_extent: u32 = 8;

// The deepest chain this can describe. A surface is bounded by the device's
// maxImageDimension2D, the base is half of that, and halving 8192 reaches the
// floor above at level ten, so eleven levels covers a surface of 16384. A device
// reporting a larger limit gets a shallower chain rather than a wrong one,
// because `chainDepth` stops counting here.
pub const max_levels: u32 = 11;

// The chain's base, which is half the target on each axis. Bloom is
// low-frequency by construction, so the finest level it needs is already coarser
// than the picture, and halving quarters the cost of every step under it.
//
// Clamped at one texel: a target one texel wide still has a base, and a zero
// extent is not an image.
pub fn baseExtent(target: vk.Extent2D) vk.Extent2D {
    return .{
        .width = @max(target.width / 2, 1),
        .height = @max(target.height / 2, 1),
    };
}

// How many levels a base of this size supports. At least one, so a base too
// small for any reduction is a chain of the base alone rather than an empty one.
pub fn chainDepth(base: vk.Extent2D) u32 {
    var depth: u32 = 1;
    while (depth < max_levels) : (depth += 1) {
        if (image.mipExtent(base.width, depth) < min_level_extent or
            image.mipExtent(base.height, depth) < min_level_extent) break;
    }
    return depth;
}

// The look, supplied per recording by composition. Not retained by the pass, for
// the reason `post.Settings` is not: what has already been submitted must not
// change because a later frame asked for something else.
pub const Settings = struct {
    // Where a pixel starts contributing, in the units the target holds, where
    // one is the radiance that will be displayed as white.
    //
    // One, and it is not an arbitrary preference. Khronos publishes a bloom
    // reference for EmissiveStrengthTest, whose five cubes carry emissive
    // [0.1, 0.5, 0.9] at strengths 1 through 16, so their brightest channels are
    // 0.9, 1.8, 3.6, 7.2 and 14.4. In that picture the first cube has no halo at
    // all and the second has a faint one, which puts the threshold above 0.9 and
    // the ramp reaching full contribution somewhere past 1.8.
    threshold: f32 = 1,

    // How far past the threshold full contribution is reached. Two, from the
    // same picture: it is what makes the second cube's halo faint rather than
    // absent or complete.
    //
    // The ramp is smooth in value and slope, so a surface drifting across the
    // threshold fades in rather than switching on. A hard cutoff shows up as a
    // flicker on anything animated through that brightness.
    knee: f32 = 2,

    // The contribution every pixel makes whatever its brightness. Zero is the
    // thresholded picture. One removes the threshold entirely, which is the
    // physically honest case, and having it expressible is what makes the
    // paragraph above falsifiable rather than a preference stated once.
    minimum: f32 = 0,

    // How much of each level carries into the level above it, and so how heavy
    // the halo's tail is. Level k reaches level zero multiplied by this to the
    // k, so a small value concentrates the glow near its source and a large one
    // spreads it.
    //
    // 0.7 is a look, and no external reference ranks it. Bloom is not in
    // glTF 2.0, and the corpus's own EmissiveStrengthTest screenshot cannot
    // stand in for one: its README states it was rendered in BabylonJS with the
    // IBL strength lowered by hand and that engine's own bloom, and its cube
    // colours are those of radiance clamped rather than tone mapped.
    //
    // Godot spends the same budget as a list of per-level weights rather than a
    // decay, 0, 0.8, 0.4, 0.1 and then nothing, in `Environment::Environment`,
    // scene/resources/environment.cpp. So it gives the finest level no weight at
    // all and stops after the third, where this decays over every level the
    // chain has.
    scatter: f32 = 0.7,

    // What the composite multiplies the normalized chain by. Godot's default for
    // the same quantity, `glow_intensity` in Environment.xml.
    intensity: f32 = 0.3,
};

pub const SettingsError = error{
    InvalidThreshold,
    InvalidKnee,
    InvalidMinimum,
    InvalidScatter,
    InvalidIntensity,
};

// The validated look, together with the one value derived from it. Only
// `resolve` produces one, so a recording cannot be handed numbers nothing has
// checked: the shader divides by the knee's span and the composite divides by a
// series, and neither has a defined answer for every float.
pub const Look = struct {
    threshold: f32,
    knee: f32,
    minimum: f32,
    scatter: f32,
    // What the post pass multiplies level zero by, intensity included.
    composite: f32,
};

// Validates the look and works out what the composite owes the chain.
//
// Each upsample step scales what it adds and nothing scales what is already
// there, so level k reaches level zero multiplied by `scatter` to the k, and a
// chain of n levels sums to (1 - w^n) / (1 - w) for a field that is uniformly
// bright. Dividing by that sum is what keeps a resize from changing the look:
// a window that admits one level fewer would otherwise dim the whole glow.
//
// Scatter of one is excluded rather than special-cased. The series does not
// decay there, and the expression above is zero over zero.
pub fn resolve(settings: Settings, level_count: u32) SettingsError!Look {
    std.debug.assert(level_count >= 1 and level_count <= max_levels);

    if (!std.math.isFinite(settings.threshold) or settings.threshold < 0)
        return error.InvalidThreshold;
    // A knee of zero collapses the smooth ramp onto a step, which `smoothstep`
    // answers by dividing by the span.
    if (!std.math.isFinite(settings.knee) or settings.knee <= 0)
        return error.InvalidKnee;
    if (!std.math.isFinite(settings.minimum) or settings.minimum < 0 or settings.minimum > 1)
        return error.InvalidMinimum;
    if (!std.math.isFinite(settings.scatter) or settings.scatter < 0 or settings.scatter >= 1)
        return error.InvalidScatter;
    if (!std.math.isFinite(settings.intensity) or settings.intensity < 0)
        return error.InvalidIntensity;

    const w = settings.scatter;
    const series = (1 - std.math.pow(f32, w, @floatFromInt(level_count))) / (1 - w);

    return .{
        .threshold = settings.threshold,
        .knee = settings.knee,
        .minimum = settings.minimum,
        .scatter = w,
        .composite = settings.intensity / series,
    };
}

// The push block the chain's shader is created against, and the record path
// fills. A supplied shader declares this layout or it is created against a
// range that does not describe it.
//
// Both fragment entry points read the whole block, each ignoring the fields the
// other needs, because both pipelines are built against one layout.
pub const PushConstants = extern struct {
    source_texel: [2]f32,
    exposure: f32,
    threshold: f32,
    knee: f32,
    minimum: f32,
    weight: f32,
    // Whether this is the step reading the HDR target, which is the only one
    // that exposes, weights by luminance and thresholds.
    extract: u32,
};

pub const push_constant_range: vk.PushConstantRange = .{
    .stage_flags = .{ .fragment_bit = true },
    .offset = 0,
    .size = @sizeOf(PushConstants),
};

// One over an extent, which is what the shader offsets its taps by. Taken from
// the extent the pass is about to set as the viewport, so the kernel cannot be
// spaced for a level other than the one being read.
pub fn texelSize(extent: vk.Extent2D) [2]f32 {
    return .{
        1 / @as(f32, @floatFromInt(extent.width)),
        1 / @as(f32, @floatFromInt(extent.height)),
    };
}

// The whole descriptor interface a chain shader may read. One binding, and a
// set per step: the source changes every draw while nothing else in the layout
// does.
pub const bindings = [_]descriptors.Binding{
    .{ .slot = 0, .name = "source", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
};

pub const Sets = descriptors.Sets(&bindings);

// Set zero reads the HDR target and set k + 1 reads level k. Written once per
// chain rather than per frame, so the recording binds a set and pushes.
//
// Sized for the deepest chain rather than for the current one. A resize changes
// how many levels there are, and a pool that had to be reallocated to match
// would put a descriptor pool allocation on the resize path for nothing: the
// sets past the chain's depth are never written and never bound.
const set_count: u32 = max_levels + 1;

fn sourceSet(step: u32) u32 {
    return step;
}

fn levelSet(level: u32) u32 {
    return level + 1;
}

// The same pair the HDR target carries, and for the same reason: every level is
// rendered into and then read by the step above it or by the post pass.
const chain_usage: vk.ImageUsageFlags = .{ .color_attachment_bit = true, .sampled_bit = true };

// The three vertices of the covering triangle, generated from the vertex index.
pub const vertex_count: u32 = 3;

pub const InitError = image.InitError ||
    image.MipViewError ||
    descriptors.InitError ||
    pipeline.CreateError ||
    vk.DeviceWrapper.CreateSamplerError;

// What a chain shader has to supply for this pass to be built from it.
//
// One set of words carrying three entry points, and the two directions share
// the vertex stage. That is not an economy: they share the binding and the push
// block as well, so one pipeline layout serves both and one descriptor set is
// rebound per step. Words that declared two vertex stages would still work and
// would say that the directions differ by more than they do.
//
// The two fragment stages are named for what they do here rather than as a
// stage and its variant. The generic spelling belonged to a table that served
// every pass at once; this one serves the chain, and which direction a pipeline
// is built for is the only thing a caller can get wrong.
//
// The interface the words have to honour is one sampled source at set 0 slot 0
// and the push block `PushConstants` mirrors. Anything further cannot be
// created against the layout `init` builds.
pub const Shader = struct {
    spirv: []const u32,
    vertex_entry: [*:0]const u8,
    downsample_entry: [*:0]const u8,
    upsample_entry: [*:0]const u8,
};

// The downsample's pipeline. Solid: it writes every texel of its target and has
// no depth attachment to test against, so the mode contributes only a disabled
// blend.
pub fn downConfig(
    layout: vk.PipelineLayout,
    module: vk.ShaderModule,
    shader: Shader,
    format: vk.Format,
) pipeline.Config {
    return .{
        .mode = .solid,
        .streams = null,
        .culling = .{ .fixed = .{} },
        .formats = .{ .colour = format },
        .layout = layout,
        .stages = .{
            .vertex = .{ .module = module, .entry_point = shader.vertex_entry },
            .fragment = .{ .module = module, .entry_point = shader.downsample_entry },
        },
    };
}

// The upsample's. Additive, which is the whole difference: its target already
// holds that level's own reduction, and this draw is a summand.
pub fn upConfig(
    layout: vk.PipelineLayout,
    module: vk.ShaderModule,
    shader: Shader,
    format: vk.Format,
) pipeline.Config {
    return .{
        .mode = .additive,
        .streams = null,
        .culling = .{ .fixed = .{} },
        .formats = .{ .colour = format },
        .layout = layout,
        .stages = .{
            .vertex = .{ .module = module, .entry_point = shader.vertex_entry },
            .fragment = .{ .module = module, .entry_point = shader.upsample_entry },
        },
    };
}

pub const BloomPass = struct {
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    // The one the descriptor sets were allocated from, kept so `deinit` returns
    // them to it rather than to whichever allocator is in reach.
    allocator: Allocator,

    // Persistent across a resize: nothing in them depends on the extent.
    sampler: vk.Sampler,
    module: vk.ShaderModule,
    layout: vk.PipelineLayout,
    down_pipeline: vk.Pipeline,
    up_pipeline: vk.Pipeline,
    sets: Sets,
    format: vk.Format,

    // Replaced by a resize. One image of `level_count` levels, plus a view of
    // each level: the chain is read one level at a time and written one level at
    // a time, and the image's own view spans all of them, which is the one thing
    // no step here wants.
    chain: image.Image,
    level_views: [max_levels]vk.ImageView,
    level_count: u32,
    // What the first downsample reads, kept rather than passed per recording so
    // that the extent the chain was built for and the extent its taps are spaced
    // for cannot come from two places.
    source_extent: vk.Extent2D,

    // The format is the HDR target's, taken rather than asked for again: the
    // pipelines are built once and a chain in another format would not be
    // renderable through them.
    pub fn init(
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        allocator: Allocator,
        extent: vk.Extent2D,
        format: vk.Format,
        source_view: vk.ImageView,
        shader: Shader,
    ) InitError!BloomPass {
        const sampler = try createSampler(context);
        errdefer context.device.destroySampler(sampler, null);

        const module = try pipeline.createModule(context, shader.spirv);
        errdefer context.device.destroyShaderModule(module, null);

        var sets = try Sets.init(context, allocator, set_count);
        errdefer sets.deinit(context, allocator);

        const layout = try pipeline.createLayout(context, .{
            .descriptor_sets = &.{sets.layout},
            .push_constants = &.{push_constant_range},
        });
        errdefer context.device.destroyPipelineLayout(layout, null);

        const down_pipeline = try pipeline.create(context, downConfig(layout, module, shader, format));
        errdefer context.device.destroyPipeline(down_pipeline, null);

        const up_pipeline = try pipeline.create(context, upConfig(layout, module, shader, format));
        errdefer context.device.destroyPipeline(up_pipeline, null);

        var self: BloomPass = .{
            .context = context,
            .memory_allocator = memory_allocator,
            .allocator = allocator,
            .sampler = sampler,
            .module = module,
            .layout = layout,
            .down_pipeline = down_pipeline,
            .up_pipeline = up_pipeline,
            .sets = sets,
            .format = format,
            .chain = undefined,
            .level_views = undefined,
            .level_count = 0,
            .source_extent = extent,
        };
        try self.createChain(extent, source_view);
        return self;
    }

    // Vulkan specification, vkDestroyPipeline and the rest: every submission
    // naming any of these must have completed. The caller drains the device.
    pub fn deinit(self: *BloomPass) void {
        const device = self.context.device;
        self.destroyChain();
        device.destroyPipeline(self.up_pipeline, null);
        device.destroyPipeline(self.down_pipeline, null);
        device.destroyPipelineLayout(self.layout, null);
        self.sets.deinit(self.context, self.allocator);
        device.destroyShaderModule(self.module, null);
        device.destroySampler(self.sampler, null);
        self.* = undefined;
    }

    // Rebuild the chain for a new target. The pipelines, the layout, the sets and
    // the sampler survive, so a resize costs one image and its views.
    //
    // Candidate first, as `Renderer.resize` is: the old chain is destroyed only
    // once the new one exists, so a resize that cannot be honoured leaves a pass
    // that still draws.
    pub fn recreate(self: *BloomPass, extent: vk.Extent2D, source_view: vk.ImageView) InitError!void {
        var previous_chain = self.chain;
        const previous_views = self.level_views;
        const previous_count = self.level_count;

        self.createChain(extent, source_view) catch |err| {
            self.chain = previous_chain;
            self.level_views = previous_views;
            self.level_count = previous_count;
            return err;
        };

        for (previous_views[0..previous_count]) |view|
            self.context.device.destroyImageView(view, null);
        previous_chain.deinit();
    }

    fn createChain(self: *BloomPass, extent: vk.Extent2D, source_view: vk.ImageView) InitError!void {
        const base = baseExtent(extent);
        const levels = chainDepth(base);

        var chain = try image.Image.init(self.context, self.memory_allocator, .{
            .width = base.width,
            .height = base.height,
            .format = self.format,
            .mip_levels = levels,
            .usage = chain_usage,
            .kind = .colour,
        });
        errdefer chain.deinit();

        var views: [max_levels]vk.ImageView = undefined;
        var created: u32 = 0;
        errdefer for (views[0..created]) |view|
            self.context.device.destroyImageView(view, null);
        while (created < levels) : (created += 1)
            views[created] = try chain.createMipView(created);

        self.chain = chain;
        self.level_views = views;
        self.level_count = levels;
        self.source_extent = extent;
        self.writeSources(source_view);
    }

    fn destroyChain(self: *BloomPass) void {
        for (self.level_views[0..self.level_count]) |view|
            self.context.device.destroyImageView(view, null);
        self.chain.deinit();
    }

    // Point set zero at the HDR target and set k + 1 at level k. Safe only when
    // no frame is in flight, which is where a chain is built: init and resize.
    //
    // The sets past the chain's depth are left as they were allocated. Nothing
    // binds them, because `record` walks the levels this chain has.
    fn writeSources(self: *const BloomPass, source_view: vk.ImageView) void {
        var infos: [set_count]vk.DescriptorImageInfo = undefined;
        var writes: [set_count]vk.WriteDescriptorSet = undefined;

        const written = self.level_count + 1;
        for (infos[0..written], writes[0..written], 0..) |*info, *write, index| {
            info.* = .{
                .sampler = self.sampler,
                .image_view = if (index == 0) source_view else self.level_views[index - 1],
                // The layout every source is in when it is read: the main pass
                // leaves the target in it, and `record`'s barriers leave each
                // level in it before the step above reads it.
                .image_layout = pass.sampled_layout,
            };
            write.* = .{
                .dst_set = self.sets.set(index),
                .dst_binding = bindings[0].slot,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = bindings[0].kind,
                .p_image_info = @ptrCast(info),
                .p_buffer_info = &no_buffers,
                .p_texel_buffer_view = &no_texel_buffers,
            };
        }
        self.context.device.updateDescriptorSets(writes[0..written], null);
    }

    // How deep the chain built for the current target is, which is what
    // `resolve` needs to normalize the composite.
    pub fn levelCount(self: *const BloomPass) u32 {
        return self.level_count;
    }

    // What the post pass samples: level zero alone, through its own view rather
    // than through the image's. A view of one level is what keeps the composite
    // from depending on the layout of the levels under it, none of which the
    // final barrier touches.
    pub fn compositeView(self: *const BloomPass) vk.ImageView {
        return self.level_views[0];
    }

    pub fn compositeSampler(self: *const BloomPass) vk.Sampler {
        return self.sampler;
    }

    fn levelExtent(self: *const BloomPass, level: u32) vk.Extent2D {
        return self.chain.levelExtent(level);
    }

    // Record the whole chain, down and then up.
    //
    // The HDR target is already in the sampled layout, because the main pass
    // ended by putting it there, and every level is left in that layout too, so
    // the post pass can read level zero without a barrier of its own.
    //
    // Infallible. Everything a step indexes is bounded by `level_count`, which
    // the chain was built with, and the look was validated where it was
    // produced.
    pub fn record(
        self: *const BloomPass,
        command_buffer: vk.CommandBuffer,
        look: Look,
        // The value the post pass exposes the scene with. The chain carries it
        // so that the threshold is a statement about what will be displayed as
        // white rather than about the scene's own units.
        exposure: f32,
    ) void {
        const device = self.context.device;

        var step: u32 = 0;
        while (step < self.level_count) : (step += 1) {
            var barriers: [2]vk.ImageMemoryBarrier2 = undefined;
            var count: u32 = 1;
            barriers[0] = self.levelBarrier(step, .discard_to_attachment);
            if (step > 0) {
                barriers[1] = self.levelBarrier(step - 1, .attachment_to_sampled);
                count = 2;
            }
            device.cmdPipelineBarrier2(command_buffer, &.{
                .image_memory_barrier_count = count,
                .p_image_memory_barriers = &barriers,
            });

            const source_extent = if (step == 0)
                self.source_extent
            else
                self.levelExtent(step - 1);

            self.draw(command_buffer, .{
                .pipeline = self.down_pipeline,
                .set = sourceSet(step),
                .level = step,
                .load_op = .dont_care,
                .push = .{
                    .source_texel = texelSize(source_extent),
                    .exposure = exposure,
                    .threshold = look.threshold,
                    .knee = look.knee,
                    .minimum = look.minimum,
                    .weight = 0,
                    .extract = @intFromBool(step == 0),
                },
            });
        }

        // Up from the deepest level. The source was written by the step before
        // and needs its read barrier; the target goes back from sampled to
        // attachment, and the blend reads it, which is why that barrier's
        // destination scope carries a read as well as a write.
        var level = self.level_count;
        while (level > 1) {
            level -= 1;
            const barriers = [2]vk.ImageMemoryBarrier2{
                self.levelBarrier(level, .attachment_to_sampled),
                self.levelBarrier(level - 1, .sampled_to_attachment),
            };
            device.cmdPipelineBarrier2(command_buffer, &.{
                .image_memory_barrier_count = barriers.len,
                .p_image_memory_barriers = &barriers,
            });

            self.draw(command_buffer, .{
                .pipeline = self.up_pipeline,
                .set = levelSet(level),
                .level = level - 1,
                // The target holds its own reduction and this draw adds to it,
                // so its contents are the one thing that must survive the start
                // of the pass.
                .load_op = .load,
                .push = .{
                    .source_texel = texelSize(self.levelExtent(level)),
                    .exposure = exposure,
                    .threshold = look.threshold,
                    .knee = look.knee,
                    .minimum = look.minimum,
                    .weight = look.scatter,
                    .extract = 0,
                },
            });
        }

        // Level zero is still an attachment: either the last upsample wrote it,
        // or the chain is one level deep and the extract did. Hand it to the
        // post pass either way.
        const final = self.levelBarrier(0, .attachment_to_sampled);
        device.cmdPipelineBarrier2(command_buffer, &.{
            .image_memory_barrier_count = 1,
            .p_image_memory_barriers = @ptrCast(&final),
        });
    }

    const Step = struct {
        pipeline: vk.Pipeline,
        set: u32,
        level: u32,
        load_op: vk.AttachmentLoadOp,
        push: PushConstants,
    };

    fn draw(self: *const BloomPass, command_buffer: vk.CommandBuffer, step: Step) void {
        const device = self.context.device;
        const extent = self.levelExtent(step.level);

        const colour = vk.RenderingAttachmentInfo{
            .image_view = self.level_views[step.level],
            .image_layout = .color_attachment_optimal,
            .resolve_mode = .{},
            .resolve_image_layout = .undefined,
            .load_op = step.load_op,
            .store_op = .store,
            .clear_value = undefined,
        };
        device.cmdBeginRendering(command_buffer, &.{
            .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = extent },
            .layer_count = 1,
            .view_mask = 0,
            .color_attachment_count = 1,
            .p_color_attachments = @ptrCast(&colour),
        });

        // Set inside the rendering instance rather than once for the whole
        // chain: every level has its own extent, so this is the one piece of
        // state that changes at each of them.
        device.cmdSetViewport(command_buffer, 0, &.{pass.viewport(extent)});
        device.cmdSetScissor(command_buffer, 0, &.{pass.scissor(extent)});

        device.cmdBindPipeline(command_buffer, .graphics, step.pipeline);
        device.cmdBindDescriptorSets(
            command_buffer,
            .graphics,
            self.layout,
            0,
            &.{self.sets.set(step.set)},
            &.{},
        );
        var push = step.push;
        device.cmdPushConstants(
            command_buffer,
            self.layout,
            push_constant_range.stage_flags,
            push_constant_range.offset,
            push_constant_range.size,
            @ptrCast(&push),
        );
        device.cmdDraw(command_buffer, vertex_count, 1, 0, 0);
        device.cmdEndRendering(command_buffer);
    }

    fn levelBarrier(
        self: *const BloomPass,
        level: u32,
        transition: LevelTransition,
    ) vk.ImageMemoryBarrier2 {
        return levelBarrierFor(self.chain.handle, level, transition);
    }
};

// What one level is moving between, at the point in the chain where it moves.
pub const LevelTransition = enum {
    // The first write to this level this frame. The old layout is undefined
    // because last frame's contents are about to be replaced in full, and the
    // source scope names two things: last frame's read of this level, which
    // needs only the execution half of the dependency, and last frame's write of
    // it, which needs the access half as well.
    discard_to_attachment,

    // The level has been written and the next step, or the post pass, is about
    // to sample it.
    attachment_to_sampled,

    // The upsample's target. A write after the read the downsample above made of
    // it, so the source scope is execution only; the destination scope carries a
    // read as well as a write because the additive blend reads what is there.
    sampled_to_attachment,
};

const level_range_template = vk.ImageSubresourceRange{
    .aspect_mask = .{ .color_bit = true },
    .base_mip_level = 0,
    .level_count = 1,
    .base_array_layer = 0,
    .layer_count = 1,
};

// Split from the pass so the masks can be exercised without a device. Every one
// of them names a hazard between two steps of one command buffer, and the
// validation layer is the only other thing that reads them.
pub fn levelBarrierFor(
    handle: vk.Image,
    level: u32,
    transition: LevelTransition,
) vk.ImageMemoryBarrier2 {
    var range = level_range_template;
    range.base_mip_level = level;

    var barrier = vk.ImageMemoryBarrier2{
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
            barrier.src_stage_mask = .{ .fragment_shader_bit = true, .color_attachment_output_bit = true };
            barrier.src_access_mask = .{ .color_attachment_write_bit = true };
            barrier.dst_stage_mask = .{ .color_attachment_output_bit = true };
            barrier.dst_access_mask = .{ .color_attachment_write_bit = true };
            barrier.old_layout = .undefined;
            barrier.new_layout = .color_attachment_optimal;
        },
        .attachment_to_sampled => {
            barrier.src_stage_mask = .{ .color_attachment_output_bit = true };
            barrier.src_access_mask = .{ .color_attachment_write_bit = true };
            barrier.dst_stage_mask = .{ .fragment_shader_bit = true };
            barrier.dst_access_mask = .{ .shader_read_bit = true };
            barrier.old_layout = .color_attachment_optimal;
            barrier.new_layout = pass.sampled_layout;
        },
        .sampled_to_attachment => {
            barrier.src_stage_mask = .{ .fragment_shader_bit = true };
            barrier.dst_stage_mask = .{ .color_attachment_output_bit = true };
            barrier.dst_access_mask = .{
                .color_attachment_read_bit = true,
                .color_attachment_write_bit = true,
            };
            barrier.old_layout = pass.sampled_layout;
            barrier.new_layout = .color_attachment_optimal;
        },
    }
    return barrier;
}

// Linear and clamped to the edge. Linear is what makes the tap patterns work at
// all: each fetch stands in for four texels, which is where the thirteen taps of
// the downsample get their width from.
//
// Clamped rather than bordered, so a tap leaving a level repeats its edge texel.
// A border colour would darken every edge of the picture by pulling black into
// the blur.
//
// One level per view, so the mipmap mode never selects anything and the maximum
// level of detail is zero. Naming a chain here would be a second, disagreeing
// account of the levels this pass already walks by hand.
fn createSampler(context: *const Context) vk.DeviceWrapper.CreateSamplerError!vk.Sampler {
    return context.device.createSampler(&.{
        .mag_filter = .linear,
        .min_filter = .linear,
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
        .border_color = .float_opaque_black,
        .unnormalized_coordinates = .false,
    }, null);
}

// Vulkan specification, VkWriteDescriptorSet: the members not selected by
// descriptorType are ignored, but the pointers are not optional.
const no_buffers = [_]vk.DescriptorBufferInfo{};
const no_texel_buffers = [_]vk.BufferView{};
