const std = @import("std");
const vk = @import("vulkan");
const attachment = @import("attachment.zig");
const Context = @import("../device/context.zig").Context;
const descriptors = @import("../binding/descriptors.zig");
const image = @import("../object/image.zig");
const memory = @import("../memory/allocator.zig");
const pass = @import("scene.zig");
const pipeline = @import("../binding/pipeline.zig");
const res = @import("lenore-resources");

const Allocator = std.mem.Allocator;
const AlphaMode = res.MaterialInfo.Rendering.AlphaMode;

// The sun shadow map: one depth image the casters are rasterized into through
// the fit's orthographic matrix, and the comparison sampler the main pass reads
// it back through.
//
// What is here and what is not. This owns the image, the sampler, the descriptor
// set the main pass reads it from, and the four pipelines a caster is drawn
// with. It does not own the loop that walks the casters: that reads the same
// batch list the main pass draws from, and splitting the two loops across two
// files would put one list's ordering rules in two places. The relationship is
// `scene.zig`'s to `renderer.zig` and not `morph.zig`'s, which has a registration
// list of its own to walk.
//
// The map is a whole scene's, not a frame's. One image serves every frame in
// flight, so a bake is ordered against the previous frame's sampling by the
// barriers below, exactly as the main pass orders its shared attachments.

// The map's own descriptor set, which is set 3 and not part of the scene set.
//
// The reason is the bake. Its pipeline layout has to name sets 0 to 2 to reach
// the frame rings and the caster's base colour, so any set in that range is
// bound while the map is being written into. Keeping the map out of them means
// the image is never both an attachment and a bound descriptor, rather than
// relying on a shader not to declare what it must not read.
pub const bindings = [_]descriptors.Binding{
    .{ .slot = 0, .name = "sun_map", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
};

pub const Sets = descriptors.Sets(&bindings);

// What a masked caster needs to run the cutoff test. The bake's shader is
// created against this block and the record path fills it.
//
// Pushed rather than read out of the material array. Two floats against binding
// that array into the bake and carrying a material index through as a varying,
// and it leaves the bake independent of that array's layout.
pub const PushConstants = extern struct {
    alpha_cutoff: f32,
    factor_alpha: f32,
};

pub const push_constant_range: vk.PushConstantRange = .{
    .stage_flags = .{ .fragment_bit = true },
    .offset = 0,
    .size = @sizeOf(PushConstants),
};

// Which of the four pipelines a caster is drawn with. Both axes change what the
// device does: the vertex input and entry point follow the mesh's streams, and
// the presence of a fragment stage follows the material's alpha mode.
pub const Variant = struct {
    skinned: bool,
    masked: bool,
};

const variant_count = 4;

pub fn pipelineIndex(variant: Variant) usize {
    return @as(usize, @intFromBool(variant.skinned)) * 2 + @intFromBool(variant.masked);
}

// Whether a material's primitives are baked at all, and if so whether they are
// tested per fragment.
//
// BLEND does not cast. A surface that is partly transparent has no one depth to
// record, and the map holds exactly one value per texel, so the choice is
// between a full shadow and none. glTF 2.0 section 3.9.4 makes BLEND the mode
// for glass and foliage cards drawn as sheets, and a full shadow is the more
// wrong of the two.
//
// MASK casts through the fragment stage. Drawn solid it would cast the shadow of
// the quad a cutout is printed on rather than of the shape the cutout shows,
// which is the whole of what the mode exists to express.
pub fn casterVariant(alpha: AlphaMode, skinned: bool) ?Variant {
    return switch (alpha) {
        .blend => null,
        .mask => .{ .skinned = skinned, .masked = true },
        .@"opaque" => .{ .skinned = skinned, .masked = false },
    };
}

// The slope-scaled depth bias the casters are rasterized with.
//
// Two, and derived rather than tuned. Vulkan scales this factor by the polygon's
// maximum depth slope, which is its depth change across one texel along the
// steeper axis (see `pipeline.DepthBias` for the section). The comparison is a
// 2x2 footprint, so the texel a fragment is tested against lies up to one texel
// away along each axis, and the depth error such a displacement can produce is
// the sum over the two axes: twice the maximum slope.
//
// The constant factor stays zero, and that is not the same decision. Its unit on
// a floating-point depth attachment is defined per primitive from the exponent
// range that primitive spans, so a value tuned against one caster is a different
// offset on the next. The constant half of the defence is carried instead by the
// normal offset, which `SunShadowSettings.normalOffsetWorld` derives from the
// map's own texel size and which the lookup applies.
const depth_bias: pipeline.DepthBias = .{ .slope = 2 };

// No culling, for the corpus rather than for a preference. A glTF scene is full
// of open single-sided surfaces, cloth and leaves and shells modelled without a
// back, and culling a face class punches a hole in the shadow of every one of
// them. The pipelines are created with fixed culling, so the bake cannot inherit
// a dynamic cull mode the main pass left behind either.
const culling: pipeline.Culling = .{ .fixed = .{} };

// What a bake shader has to supply for this pass to be built from it.
//
// Three entry points against four pipelines, because the two axes are not
// symmetric. The vertex stage varies over the mesh's skinning and the fragment
// stage exists only for the masked half, so an opaque caster is built with no
// fragment stage at all rather than with one that discards nothing.
//
// The two vertex names are separate fields rather than an array over the
// skinning axis. The bake's `pipelineIndex` runs over both axes at once, so an
// array here would be indexed differently from the pipelines it feeds and would
// invite the two to be confused; the scene's four variants are the case where
// one index really does serve both.
pub const Shader = struct {
    spirv: []const u32,
    vertex_entry: [*:0]const u8,
    skinned_vertex_entry: [*:0]const u8,
    masked_fragment_entry: [*:0]const u8,
};

pub const InitError = error{
    // The requested map does not fit the device's two-dimensional image limit,
    // or has no extent at all.
    MapSizeUnsupported,
} || attachment.FormatError || image.InitError || descriptors.InitError || pipeline.CreateError;

// The three set layouts the bake's pipeline layout names, in set order. Set 1 is
// named and never bound: a bake shader declares nothing out of it, and it is
// there so that the material's texture set keeps index 2 and the binding stays
// valid across the two pipeline layouts.
pub const Layouts = struct {
    frame: vk.DescriptorSetLayout,
    scene: vk.DescriptorSetLayout,
    material: vk.DescriptorSetLayout,
};

pub const ShadowPass = struct {
    context: *const Context,
    // The one the descriptor sets were allocated from, kept so that `deinit`
    // returns them to it rather than to whichever allocator is in reach.
    allocator: Allocator,
    map: image.Image,
    sampler: vk.Sampler,
    module: vk.ShaderModule,
    layout: vk.PipelineLayout,
    pipelines: [variant_count]vk.Pipeline,
    sets: Sets,

    pub fn init(
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        allocator: Allocator,
        map_size: u32,
        layouts: Layouts,
        shader: Shader,
    ) InitError!ShadowPass {
        if (map_size == 0 or map_size > context.properties.limits.max_image_dimension_2d)
            return error.MapSizeUnsupported;

        const format = try attachment.shadowFormat(context);

        var map = try image.Image.init(context, memory_allocator, .{
            .width = map_size,
            .height = map_size,
            .format = format,
            .usage = attachment.shadow_usage,
            .kind = .depth,
        });
        errdefer map.deinit();

        const sampler = try createSampler(context, format);
        errdefer context.device.destroySampler(sampler, null);

        const module = try pipeline.createModule(context, shader.spirv);
        errdefer context.device.destroyShaderModule(module, null);

        var sets = try Sets.init(context, allocator, 1);
        errdefer sets.deinit(context, allocator);

        const layout = try pipeline.createLayout(context, .{
            .descriptor_sets = &.{ layouts.frame, layouts.scene, layouts.material },
            .push_constants = &.{push_constant_range},
        });
        errdefer context.device.destroyPipelineLayout(layout, null);

        var pipelines: [variant_count]vk.Pipeline = undefined;
        var created: usize = 0;
        errdefer for (pipelines[0..created]) |built| context.device.destroyPipeline(built, null);

        for ([_]bool{ false, true }) |skinned| {
            for ([_]bool{ false, true }) |masked| {
                // The rollback above walks the built prefix, so the loop has to
                // fill the array in index order.
                std.debug.assert(pipelineIndex(.{ .skinned = skinned, .masked = masked }) == created);
                pipelines[created] = try pipeline.create(context, .{
                    // Solid: the bake tests depth and adds to it, which is what
                    // makes the map the nearest caster rather than the last one
                    // drawn.
                    .mode = .solid,
                    .streams = .{ .skinned = skinned },
                    .culling = culling,
                    .formats = .{ .depth = format },
                    .layout = layout,
                    .depth_bias = depth_bias,
                    .stages = .{
                        .vertex = .{
                            .module = module,
                            .entry_point = if (skinned)
                                shader.skinned_vertex_entry
                            else
                                shader.vertex_entry,
                        },
                        // An opaque caster has no fragment stage at all, so
                        // nothing stands between the rasterizer and the depth
                        // write.
                        .fragment = if (masked) .{
                            .module = module,
                            .entry_point = shader.masked_fragment_entry,
                        } else null,
                    },
                });
                created += 1;
            }
        }

        var self: ShadowPass = .{
            .context = context,
            .allocator = allocator,
            .map = map,
            .sampler = sampler,
            .module = module,
            .layout = layout,
            .pipelines = pipelines,
            .sets = sets,
        };
        self.write();
        return self;
    }

    // Vulkan specification, vkDestroyPipeline and the rest: every submission
    // naming any of these must have completed. The caller drains the device.
    pub fn deinit(self: *ShadowPass) void {
        const device = self.context.device;
        for (self.pipelines) |built| device.destroyPipeline(built, null);
        device.destroyPipelineLayout(self.layout, null);
        self.sets.deinit(self.context, self.allocator);
        device.destroyShaderModule(self.module, null);
        device.destroySampler(self.sampler, null);
        self.map.deinit();
        self.* = undefined;
    }

    // Points the map's own set at the image. Once, at init: neither the image
    // nor the sampler is replaced for the life of the pass, so this is not on
    // any frame path.
    fn write(self: *const ShadowPass) void {
        const info = vk.DescriptorImageInfo{
            .sampler = self.sampler,
            .image_view = self.map.view,
            .image_layout = pass.sampled_layout,
        };
        const writes = [_]vk.WriteDescriptorSet{.{
            .dst_set = self.sets.set(0),
            .dst_binding = bindings[0].slot,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = bindings[0].kind,
            .p_image_info = @ptrCast(&info),
            .p_buffer_info = &no_buffers,
            .p_texel_buffer_view = &no_texel_buffers,
        }};
        self.context.device.updateDescriptorSets(&writes, null);
    }

    pub fn descriptorSetLayout(self: *const ShadowPass) vk.DescriptorSetLayout {
        return self.sets.layout;
    }

    pub fn descriptorSet(self: *const ShadowPass) vk.DescriptorSet {
        return self.sets.set(0);
    }

    // What the fit has to be computed against: the projection covers the whole
    // map, so the texel size and the staleness threshold both follow from this
    // number. Asked of the pass rather than carried beside it, so the two cannot
    // name different resolutions.
    pub fn mapSize(self: *const ShadowPass) u32 {
        return self.map.width;
    }

    pub fn pipelineFor(self: *const ShadowPass, variant: Variant) vk.Pipeline {
        return self.pipelines[pipelineIndex(variant)];
    }

    fn extent(self: *const ShadowPass) vk.Extent2D {
        return .{ .width = self.map.width, .height = self.map.height };
    }

    // Opens the bake. Between this and `end` the caller records casters; with no
    // caster recorded at all the map is left cleared, which reads as fully lit
    // everywhere and is the correct content for a scene that casts nothing.
    pub fn begin(self: *const ShadowPass, command_buffer: vk.CommandBuffer) void {
        const barriers = beginBarriers(self.map.handle);
        self.context.device.cmdPipelineBarrier2(command_buffer, &.{
            .image_memory_barrier_count = barriers.len,
            .p_image_memory_barriers = &barriers,
        });

        // The map's own extent, not the camera's. This is the reason the bake
        // cannot be recorded inside the main pass: they rasterize at different
        // resolutions into different attachments.
        self.context.device.cmdSetViewport(command_buffer, 0, &.{pass.viewport(self.extent())});
        self.context.device.cmdSetScissor(command_buffer, 0, &.{pass.scissor(self.extent())});

        const depth = depthAttachment(self.map.view);
        self.context.device.cmdBeginRendering(command_buffer, &.{
            .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = self.extent() },
            .layer_count = 1,
            .view_mask = 0,
            .color_attachment_count = 0,
            .p_color_attachments = &no_colour_attachments,
            .p_depth_attachment = &depth,
        });
    }

    pub fn end(self: *const ShadowPass, command_buffer: vk.CommandBuffer) void {
        self.context.device.cmdEndRendering(command_buffer);

        const barriers = endBarriers(self.map.handle);
        self.context.device.cmdPipelineBarrier2(command_buffer, &.{
            .image_memory_barrier_count = barriers.len,
            .p_image_memory_barriers = &barriers,
        });
    }
};

const depth_range = vk.ImageSubresourceRange{
    .aspect_mask = .{ .depth_bit = true },
    .base_mip_level = 0,
    .level_count = 1,
    .base_array_layer = 0,
    .layer_count = 1,
};

// One image for every frame in flight, so a bake has to be ordered after the
// previous frame's sampling of it. That is a write-after-read, and it is the
// whole reason this barrier exists; the layout change is incidental, because
// `undefined` as the old layout discards contents the clear is about to replace.
pub fn beginBarriers(handle: vk.Image) [1]vk.ImageMemoryBarrier2 {
    return .{.{
        .src_stage_mask = .{ .fragment_shader_bit = true },
        .src_access_mask = .{ .shader_read_bit = true },
        .dst_stage_mask = .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true },
        .dst_access_mask = .{ .depth_stencil_attachment_write_bit = true },
        .old_layout = .undefined,
        .new_layout = .depth_attachment_optimal,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = handle,
        .subresource_range = depth_range,
    }};
}

// Makes the bake's depth writes available to the main pass's sampler. Unlike the
// camera's depth attachment, this one is stored: what it holds is the pass's
// whole output rather than a by-product of drawing colour.
pub fn endBarriers(handle: vk.Image) [1]vk.ImageMemoryBarrier2 {
    return .{.{
        .src_stage_mask = .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true },
        .src_access_mask = .{ .depth_stencil_attachment_write_bit = true },
        .dst_stage_mask = .{ .fragment_shader_bit = true },
        .dst_access_mask = .{ .shader_read_bit = true },
        .old_layout = .depth_attachment_optimal,
        .new_layout = pass.sampled_layout,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = handle,
        .subresource_range = depth_range,
    }};
}

// Cleared to the far plane and stored. `SunShadowFit` builds its projection with
// a near of zero and a far of twice the slacked radius, and zmath's
// `orthographicRh` maps that range onto depth 0 to 1, so an untouched texel
// holds a caster infinitely far away and every comparison against it passes.
pub fn depthAttachment(view: vk.ImageView) vk.RenderingAttachmentInfo {
    return .{
        .image_view = view,
        .image_layout = .depth_attachment_optimal,
        .resolve_mode = .{},
        .resolve_image_layout = .undefined,
        .load_op = .clear,
        .store_op = .store,
        .clear_value = .{ .depth_stencil = .{ .depth = 1, .stencil = 0 } },
    };
}

// The comparison sampler the main pass reads the map through.
//
// `compare_enable` is what makes a fetch return the result of a test rather than
// a depth, and with a linear filter the four texels of the footprint are each
// tested and the results averaged, which is one tap of percentage-closer
// filtering for the cost of one fetch.
//
// The comparison is `less_or_equal` because the reference is the shaded point's
// own depth and the stored value is the nearest caster's: the point is lit when
// it is not behind that caster.
//
// Outside the fit there is no information, and the border is what a lookup lands
// on there. Opaque white is depth 1, the far plane, so every reference compares
// as lit rather than as shadowed by an empty texel.
fn createSampler(context: *const Context, format: vk.Format) vk.DeviceWrapper.CreateSamplerError!vk.Sampler {
    // A device that cannot filter this format linearly gets a single test
    // instead of an averaged four. Asked rather than assumed: the alternative is
    // a sampler the validation layer rejects on a device that lacks it.
    const features = attachment.formatFeatures(context, format);
    const filter: vk.Filter = if (features.sampled_image_filter_linear_bit) .linear else .nearest;

    return context.device.createSampler(&.{
        .mag_filter = filter,
        .min_filter = filter,
        // One level. The map is rasterized at one resolution and read at one,
        // and a chain would need a filter that has no meaning over depths.
        .mipmap_mode = .nearest,
        .address_mode_u = .clamp_to_border,
        .address_mode_v = .clamp_to_border,
        .address_mode_w = .clamp_to_border,
        .mip_lod_bias = 0,
        .anisotropy_enable = .false,
        .max_anisotropy = 1,
        .compare_enable = .true,
        .compare_op = .less_or_equal,
        .min_lod = 0,
        .max_lod = 0,
        .border_color = .float_opaque_white,
        .unnormalized_coordinates = .false,
    }, null);
}

// Vulkan specification, VkRenderingInfo and VkWriteDescriptorSet: the members
// not selected are ignored, but the pointers are not optional.
const no_colour_attachments = [_]vk.RenderingAttachmentInfo{};
const no_buffers = [_]vk.DescriptorBufferInfo{};
const no_texel_buffers = [_]vk.BufferView{};
