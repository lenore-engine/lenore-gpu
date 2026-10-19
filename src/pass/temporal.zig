const std = @import("std");
const vk = @import("vulkan");

const Context = @import("../device/context.zig").Context;
const descriptors = @import("../binding/descriptors.zig");
const image = @import("../object/image.zig");
const memory = @import("../memory/allocator.zig");
const pass = @import("scene.zig");
const pipeline = @import("../binding/pipeline.zig");

const Allocator = std.mem.Allocator;

// The temporal resolve: the main pass's jittered frame at the render
// resolution gathered with the frames before it into one picture at the output
// resolution, which bloom, the meter and the post pass then read in place of
// the main pass's target.
//
// One covering triangle per frame, after the main pass. It reads the frame's
// colour, motion and depth and last frame's resolved picture, and writes two
// attachments with the same value: `resolved`, which the passes after it read
// through descriptors written once, and one of the two history images, which
// the next frame reads. The history alternates between the two so that no draw
// reads the image it writes; `resolved` does not, so nothing downstream has to
// follow the alternation.

// What the resolve writes and keeps: the main pass's own format, read by the
// same passes that would otherwise read the main pass's target.
pub const Formats = struct {
    colour: vk.Format,
};

const usage: vk.ImageUsageFlags = .{ .color_attachment_bit = true, .sampled_bit = true };

// Who fills `resolved`. `external` is an upscaler outside this module writing it
// from compute through a storage view, in place of `record`: the image is
// created with storage usage, and the passes after read it as they always do.
// The resolve's own pipeline and histories are still built and stay unused.
pub const Writer = enum { draw, external };

// Mirrors `TemporalPushConstants` in temporal.slang.
pub const PushConstants = extern struct {
    // The render target's and the output's extents, in pixels.
    render_size: [2]f32,
    output_size: [2]f32,
    // The frame's sub-pixel offset in render pixels, x right and y down: where
    // a render texel's sample sits relative to its centre is minus this.
    jitter: [2]f32,
    // How much of the new frame each output pixel takes when a sample of this
    // frame lands on its centre.
    blend: f32,
    // Nonzero takes the frame alone: there is no history to gather with.
    reset: u32,
};

pub const push_constant_range: vk.PushConstantRange = .{
    .stage_flags = .{ .fragment_bit = true },
    .offset = 0,
    .size = @sizeOf(PushConstants),
};

// What the frame decides.
pub const Frame = struct {
    // `Camera.jitter`'s offset, in the framebuffer's device coordinates.
    jitter: [2]f32,
};

// The new frame's weight at a pixel one of its samples lands on. A tenth
// gathers about ten frames; judged on Reclaim at three quarters of 2560x1600,
// in motion, with no loss of detail seen.
pub const blend: f32 = 0.1;

// Device coordinates span two across the target, so one of their units is
// half the target's pixels.
pub fn jitterPixels(device: [2]f32, render: vk.Extent2D) [2]f32 {
    return .{
        device[0] * @as(f32, @floatFromInt(render.width)) * 0.5,
        device[1] * @as(f32, @floatFromInt(render.height)) * 0.5,
    };
}

pub fn pushFor(frame: Frame, render: vk.Extent2D, output: vk.Extent2D, reset: bool) PushConstants {
    return .{
        .render_size = .{ @floatFromInt(render.width), @floatFromInt(render.height) },
        .output_size = .{ @floatFromInt(output.width), @floatFromInt(output.height) },
        .jitter = jitterPixels(frame.jitter, render),
        .blend = blend,
        .reset = @intFromBool(reset),
    };
}

pub const bindings = [_]descriptors.Binding{
    .{ .slot = 0, .name = "current", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
    .{ .slot = 1, .name = "motion_vectors", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
    .{ .slot = 2, .name = "depth_map", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
    .{ .slot = 3, .name = "history", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
};

pub const Sets = descriptors.Sets(&bindings);

// Set k reads history k and the frame writes history 1 - k.
const set_count: u32 = 2;

// Which history a frame reads, by how many frames have been resolved since the
// history was last reset; the other one is written.
pub fn readIndex(resolved_frames: u64) u32 {
    return @intCast(resolved_frames % 2);
}

pub const vertex_count: u32 = 3;

pub const InitError = image.InitError ||
    descriptors.InitError ||
    pipeline.CreateError ||
    vk.DeviceWrapper.CreateSamplerError;

pub const Shader = struct {
    spirv: []const u32,
    vertex_entry: [*:0]const u8,
    fragment_entry: [*:0]const u8,
};

// The images a resolve reads from the main pass, which the renderer owns.
pub const Sources = struct {
    colour: vk.ImageView,
    motion: vk.ImageView,
    depth: vk.ImageView,
};

pub const TemporalPass = struct {
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    allocator: Allocator,
    format: vk.Format,
    writer: Writer,

    // Persistent across a resize.
    nearest_sampler: vk.Sampler,
    linear_sampler: vk.Sampler,
    module: vk.ShaderModule,
    layout: vk.PipelineLayout,
    pipeline: vk.Pipeline,
    sets: Sets,

    // Replaced by a resize, at the output's extent.
    resolved: image.Image,
    histories: [2]image.Image,

    // Frames resolved since the history was last reset. Zero means the history
    // holds nothing the next frame may read, and the frame is taken alone.
    resolved_frames: u64,
    // Whether the histories have been put in the sampled layout their
    // descriptors name. A new image's layout is undefined, and a descriptor
    // statically used by the pipeline must name a layout the image is in even
    // on the frame its value is not read.
    histories_ready: bool,

    pub fn init(
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        allocator: Allocator,
        format: vk.Format,
        output: vk.Extent2D,
        sources: Sources,
        shader: Shader,
        writer: Writer,
    ) InitError!TemporalPass {
        const nearest_sampler = try createSampler(context, .nearest);
        errdefer context.device.destroySampler(nearest_sampler, null);
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

        const built = try pipeline.create(context, .{
            .mode = .solid,
            .vertex_input = .none,
            .culling = .{ .fixed = .{} },
            .formats = .{ .colour = format, .colour_count = 2 },
            .layout = layout,
            .stages = .{
                .vertex = .{ .module = module, .entry_point = shader.vertex_entry },
                .fragment = .{ .module = module, .entry_point = shader.fragment_entry },
            },
        });
        errdefer context.device.destroyPipeline(built, null);

        var self: TemporalPass = .{
            .context = context,
            .memory_allocator = memory_allocator,
            .allocator = allocator,
            .format = format,
            .writer = writer,
            .nearest_sampler = nearest_sampler,
            .linear_sampler = linear_sampler,
            .module = module,
            .layout = layout,
            .pipeline = built,
            .sets = sets,
            .resolved = undefined,
            .histories = undefined,
            .resolved_frames = 0,
            .histories_ready = false,
        };
        try self.createImages(output, sources);
        return self;
    }

    // Vulkan specification, vkDestroyPipeline and the rest: every submission
    // naming any of these must have completed. The caller drains the device.
    pub fn deinit(self: *TemporalPass) void {
        const device = self.context.device;
        self.destroyImages();
        device.destroyPipeline(self.pipeline, null);
        device.destroyPipelineLayout(self.layout, null);
        self.sets.deinit(self.context, self.allocator);
        device.destroyShaderModule(self.module, null);
        device.destroySampler(self.linear_sampler, null);
        device.destroySampler(self.nearest_sampler, null);
        self.* = undefined;
    }

    // New images for a new output and new sources, and a history reset: what
    // the old images held was gathered at another size. Candidate first, so a
    // resize that cannot be honoured leaves a pass that still draws.
    pub fn recreate(self: *TemporalPass, output: vk.Extent2D, sources: Sources) InitError!void {
        const previous = self.*;
        self.createImages(output, sources) catch |err| {
            self.* = previous;
            return err;
        };
        var old = previous;
        old.destroyImages();
    }

    // The next frame takes its own picture alone, as after a camera cut.
    pub fn resetHistory(self: *TemporalPass) void {
        self.resolved_frames = 0;
    }

    fn createImages(self: *TemporalPass, output: vk.Extent2D, sources: Sources) InitError!void {
        var written_usage = usage;
        written_usage.storage_bit = self.writer == .external;
        var resolved = try self.createImage(output, written_usage);
        errdefer resolved.deinit();
        var first = try self.createImage(output, usage);
        errdefer first.deinit();
        const second = try self.createImage(output, usage);

        self.resolved = resolved;
        self.histories = .{ first, second };
        self.resolved_frames = 0;
        self.histories_ready = false;
        self.writeSets(sources);
    }

    fn createImage(self: *const TemporalPass, extent: vk.Extent2D, image_usage: vk.ImageUsageFlags) image.InitError!image.Image {
        return image.Image.init(self.context, self.memory_allocator, .{
            .width = extent.width,
            .height = extent.height,
            .format = self.format,
            .usage = image_usage,
            .kind = .colour,
        });
    }

    fn destroyImages(self: *TemporalPass) void {
        self.histories[1].deinit();
        self.histories[0].deinit();
        self.resolved.deinit();
    }

    // Safe only when no frame is in flight, which is where images are built.
    fn writeSets(self: *const TemporalPass, sources: Sources) void {
        var infos: [set_count * bindings.len]vk.DescriptorImageInfo = undefined;
        var writes: [set_count * bindings.len]vk.WriteDescriptorSet = undefined;
        for (0..set_count) |set| {
            const views = [bindings.len]vk.DescriptorImageInfo{
                .{ .sampler = self.nearest_sampler, .image_view = sources.colour, .image_layout = pass.sampled_layout },
                .{ .sampler = self.nearest_sampler, .image_view = sources.motion, .image_layout = pass.sampled_layout },
                .{ .sampler = self.nearest_sampler, .image_view = sources.depth, .image_layout = pass.sampled_layout },
                .{ .sampler = self.linear_sampler, .image_view = self.histories[set].view, .image_layout = pass.sampled_layout },
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
        self.context.device.updateDescriptorSets(&writes, null);
    }

    // What bloom, the meter and the post pass read in place of the main pass's
    // target.
    pub fn resolvedView(self: *const TemporalPass) vk.ImageView {
        return self.resolved.view;
    }

    // The image an external writer fills. Its view serves as the storage view
    // too: one level, and the image carries storage usage in that mode.
    pub fn resolvedImage(self: *const TemporalPass) *const image.Image {
        return &self.resolved;
    }

    pub fn outputExtent(self: *const TemporalPass) vk.Extent2D {
        return .{ .width = self.resolved.width, .height = self.resolved.height };
    }

    // After the main pass, whose end barrier leaves its colour and depth
    // sampled, and after the prepass's, which leaves the motion sampled. On
    // exit `resolved` and the history just written are sampled, for the passes
    // after this one and for the next frame.
    pub fn record(self: *TemporalPass, command_buffer: vk.CommandBuffer, render: vk.Extent2D, frame: Frame) void {
        const device = self.context.device;
        const output = self.outputExtent();
        const read = readIndex(self.resolved_frames);
        const written = 1 - read;

        var opening: [4]vk.ImageMemoryBarrier2 = undefined;
        var count: u32 = 0;
        opening[count] = colourBarrier(self.resolved.handle, .discard_to_attachment);
        count += 1;
        opening[count] = colourBarrier(self.histories[written].handle, .discard_to_attachment);
        count += 1;
        if (!self.histories_ready) {
            opening[count] = colourBarrier(self.histories[read].handle, .undefined_to_sampled);
            count += 1;
            self.histories_ready = true;
        }
        device.cmdPipelineBarrier2(command_buffer, &.{
            .image_memory_barrier_count = count,
            .p_image_memory_barriers = &opening,
        });

        const colours = [2]vk.RenderingAttachmentInfo{
            attachmentFor(self.resolved.view),
            attachmentFor(self.histories[written].view),
        };
        device.cmdBeginRendering(command_buffer, &.{
            .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = output },
            .layer_count = 1,
            .view_mask = 0,
            .color_attachment_count = colours.len,
            .p_color_attachments = &colours,
        });
        device.cmdSetViewport(command_buffer, 0, &.{pass.viewport(output)});
        device.cmdSetScissor(command_buffer, 0, &.{pass.scissor(output)});
        device.cmdBindPipeline(command_buffer, .graphics, self.pipeline);
        device.cmdBindDescriptorSets(command_buffer, .graphics, self.layout, 0, &.{self.sets.set(read)}, &.{});
        var block = pushFor(frame, render, output, self.resolved_frames == 0);
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

        const closing = [2]vk.ImageMemoryBarrier2{
            colourBarrier(self.resolved.handle, .attachment_to_sampled),
            colourBarrier(self.histories[written].handle, .attachment_to_sampled),
        };
        device.cmdPipelineBarrier2(command_buffer, &.{
            .image_memory_barrier_count = closing.len,
            .p_image_memory_barriers = &closing,
        });
        self.resolved_frames += 1;
    }
};

fn attachmentFor(view: vk.ImageView) vk.RenderingAttachmentInfo {
    return .{
        .image_view = view,
        .image_layout = .color_attachment_optimal,
        .resolve_mode = .{},
        .resolve_image_layout = .undefined,
        // Every pixel is written by the covering triangle.
        .load_op = .dont_care,
        .store_op = .store,
        .clear_value = undefined,
    };
}

const colour_range = vk.ImageSubresourceRange{
    .aspect_mask = .{ .color_bit = true },
    .base_mip_level = 0,
    .level_count = 1,
    .base_array_layer = 0,
    .layer_count = 1,
};

pub const Transition = enum {
    // The first write this frame. Undefined as the old layout discards what
    // the draw replaces; the source scope is the last reads of the image,
    // fragment (the resolve, bloom, the post pass) and compute (the meter).
    discard_to_attachment,
    attachment_to_sampled,
    // A history no frame has written, put in the layout its descriptor names.
    undefined_to_sampled,
};

pub fn colourBarrier(handle: vk.Image, transition: Transition) vk.ImageMemoryBarrier2 {
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
        .subresource_range = colour_range,
    };
    switch (transition) {
        .discard_to_attachment => {
            barrier.src_stage_mask = .{ .fragment_shader_bit = true, .compute_shader_bit = true };
            barrier.src_access_mask = .{ .shader_sampled_read_bit = true };
            barrier.dst_stage_mask = .{ .color_attachment_output_bit = true };
            barrier.dst_access_mask = .{ .color_attachment_write_bit = true };
            barrier.old_layout = .undefined;
            barrier.new_layout = .color_attachment_optimal;
        },
        .attachment_to_sampled => {
            barrier.src_stage_mask = .{ .color_attachment_output_bit = true };
            barrier.src_access_mask = .{ .color_attachment_write_bit = true };
            barrier.dst_stage_mask = .{ .fragment_shader_bit = true, .compute_shader_bit = true };
            barrier.dst_access_mask = .{ .shader_sampled_read_bit = true };
            barrier.old_layout = .color_attachment_optimal;
            barrier.new_layout = pass.sampled_layout;
        },
        .undefined_to_sampled => {
            barrier.dst_stage_mask = .{ .fragment_shader_bit = true };
            barrier.dst_access_mask = .{ .shader_sampled_read_bit = true };
            barrier.old_layout = .undefined;
            barrier.new_layout = pass.sampled_layout;
        },
    }
    return barrier;
}

// Clamped to the edge, so a fetch past the border repeats it rather than
// reading a border colour as a colour or a depth.
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
        .border_color = .float_opaque_black,
        .unnormalized_coordinates = .false,
    }, null);
}

// Vulkan specification, VkWriteDescriptorSet: the members not selected by
// descriptorType are ignored, but the pointers are not optional.
const no_buffers = [_]vk.DescriptorBufferInfo{};
const no_texel_buffers = [_]vk.BufferView{};
