const std = @import("std");
const vk = @import("vulkan");
const attachment = @import("attachment.zig");
const Context = @import("context.zig").Context;
const descriptors = @import("descriptors.zig");
const environment = @import("environment.zig");
const frame_set = @import("frame_set.zig");
const image = @import("image.zig");
const material_storage = @import("material_storage.zig");
const memory = @import("memory/allocator.zig");
const mesh_module = @import("mesh/resource.zig");
const pass = @import("pass.zig");
const pipeline = @import("pipeline.zig");
const post = @import("post.zig");
const res = @import("lenore-resources");
const resource_storage = @import("resource_storage.zig");
const shaders = @import("shaders.zig");
const uniforms = @import("uniforms.zig");

const Allocator = std.mem.Allocator;

// The main pass and the pass that presents its result, and everything they
// need that outlives a frame.
//
// What it does not own: meshes, textures and materials, which belong to the
// storage the engine fills, and the swapchain, which the frame loop drives. It
// is handed the image to present into.

// Which set each layout occupies, in the order the pipeline layout names them
// and the order scene.slang gives as a binding's space. They are in increasing
// order of how often the recorder binds them: the frame set once, the scene set
// once, a material set whenever the ordered list reaches a new material.
pub const frame_set_index = 0;
pub const scene_set_index = 1;
pub const material_set_index = 2;

// The whole scene set, assembled from the two files that write into it: the
// packed material array every fragment indexes, then the environment every
// fragment lights itself from. Neither list belongs in the other's file, and
// joining them here is what makes a slot claimed twice a compile error rather
// than a descriptor quietly overwritten at run time.
pub const scene_bindings = material_storage.bindings ++ environment.bindings;

const SceneSets = descriptors.Sets(&scene_bindings);

// What a material set holds: base colour, metallic-roughness, normal, emissive,
// then occlusion. One binding per slot of `resource_storage.TextureSet`, in that
// type's field order, which is also the order `material_storage.TextureSlot`
// indexes the packed transform array by.
pub const material_bindings = [_]descriptors.Binding{
    .{ .slot = 0, .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
    .{ .slot = 1, .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
    .{ .slot = 2, .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
    .{ .slot = 3, .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
    .{ .slot = 4, .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
};

comptime {
    // The layout is one binding per texture-set slot. A slot added to the set
    // without a binding here writes fewer descriptors than the shader samples,
    // and the set layout would still be created without complaint.
    std.debug.assert(
        material_bindings.len == @typeInfo(resource_storage.TextureSet).@"struct".fields.len,
    );
}

const MaterialSets = descriptors.Sets(&material_bindings);

pub const InitError = error{
    NoSupportedDepthFormat,
    NoSupportedColourFormat,
} || image.InitError ||
    frame_set.InitError ||
    descriptors.InitError ||
    pipeline.CreateError;

pub const MaterialError = error{MaterialIndexOutOfRange};
pub const FrameError = error{FrameIndexOutOfRange};
pub const UpdateError = frame_set.UpdateError || FrameError;
pub const RecordError = MaterialError || FrameError || post.SettingsError || error{
    MaterialNotConfigured,
    // The scene set has no material buffer behind it. A precondition of the
    // whole list rather than of one batch: every fragment reads that array,
    // whichever material a batch names.
    MaterialBufferNotConfigured,
    // The scene set has no environment behind it. Like the material buffer, a
    // precondition of the whole list: every fragment samples the environment
    // whatever material its batch names, and a descriptor that was never
    // written is not a dark picture but undefined behaviour.
    EnvironmentNotConfigured,
    EmptyBatch,
    InstanceRangeOutOfBounds,
    UnsupportedCullMode,
    // The list is not partitioned by layer. Ordering is the scene's to decide
    // and the recorder's to rely on, so this reports a plan that cannot be
    // composited rather than drawing it wrong.
    SolidBatchAfterBlended,
};

pub fn validateMaterialIndex(material_count: usize, index: u32) MaterialError!void {
    if (index >= material_count) return error.MaterialIndexOutOfRange;
}

pub fn validateFrameIndex(frame_count: usize, index: usize) FrameError!void {
    if (index >= frame_count) return error.FrameIndexOutOfRange;
}

// Validates every value the recorder uses as an unchecked index or range. This
// runs over the complete list before rendering begins, so a bad batch cannot
// leave an open pass or a partially recorded frame.
pub fn validateRecordBatch(
    material_modes: []const ?pipeline.Mode,
    live_instances: usize,
    material_index: u32,
    cull_mode: vk.CullModeFlags,
    first_instance: u32,
    instance_count: u32,
    // Whether a blended batch has already been recorded in this list. Threading
    // it through keeps the ordering rule in the same validation the recorder
    // already makes, and keeps it reachable from a test: everything here is a
    // scalar or a slice, so nothing needs a mesh or a device.
    blended_seen: bool,
) RecordError!void {
    try validateMaterialIndex(material_modes.len, material_index);
    const mode = material_modes[material_index] orelse return error.MaterialNotConfigured;
    // Blending composites over what is already in the attachment, so everything
    // a blended surface is meant to show through has to be drawn first. A solid
    // batch after a blended one would be composited over instead, and it also
    // writes depth, which would then reject the blended fragments already there.
    if (mode == .solid and blended_seen) return error.SolidBatchAfterBlended;
    if (instance_count == 0) return error.EmptyBatch;
    if (cull_mode.front_bit and cull_mode.back_bit) return error.UnsupportedCullMode;

    const first: usize = first_instance;
    const count: usize = instance_count;
    if (first > live_instances or count > live_instances - first)
        return error.InstanceRangeOutOfBounds;
}

// Which of the scene pipelines a mesh draws through.
pub const SceneVariant = enum { unskinned, skinned };

// The mesh's own streams decide it, not a flag the caller repeats: a pipeline's
// vertex input has to declare exactly the bindings `Mesh.bind` binds, and both
// read the same `streams`.
//
// Split from the pipeline it selects for the reason `frame_set.validate` is
// split from the write it guards: the choice is a function of the mesh alone,
// and a device is what stands between it and a test. Selecting the unskinned
// pipeline for a skinned mesh is a vertex input that declares no binding 1
// while the shader reads none either, so it draws the bind pose and reports
// nothing.
//
// Only skinning has a shader path. A mesh carrying the colour or second-UV
// stream draws unskinned, and the buffers `Mesh.bind` bound for those go
// unread.
pub fn sceneVariantFor(streams: res.VertexStreams) SceneVariant {
    return if (streams.skinned) .skinned else .unskinned;
}

// The scene pipelines, one per vertex variant and blend mode. Four rather than
// two named fields, because both axes now change what a draw does: the variant
// picks the vertex entry point and its input, and the mode decides whether the
// draw blends and whether it writes depth.
//
// It is not keyed by `streams.index()`. The colour and second-UV streams have no
// path in the shader, so a table over those would build pipelines that differ in
// nothing a draw can reach.
const scene_modes = @typeInfo(pipeline.Mode).@"enum".fields.len;
const scene_variants = @typeInfo(SceneVariant).@"enum".fields.len;
const scene_pipeline_count = scene_variants * scene_modes;

fn scenePipelineIndex(variant: SceneVariant, mode: pipeline.Mode) usize {
    return @intFromEnum(variant) * scene_modes + @intFromEnum(mode);
}

// The Vulkan-facing form the umbrella translates a scene batch into. Keeping
// this distinct from the scene plan is deliberate: the GPU module owns resource
// pointers and Vulkan cull state, while the scene module owns ordering policy.
// Composition performs that translation into preallocated storage explicitly.
pub const RecordBatch = struct {
    mesh: *const mesh_module.Mesh,
    material_index: u32,
    cull_mode: vk.CullModeFlags,
    first_instance: u32,
    instance_count: u32,
    // Replaces binding zero, for a draw whose vertices a prepass produced. Null
    // is the mesh's own buffer. Every other stream still comes from the mesh:
    // the substitute has the same layout, so the pipeline is unchanged.
    vertex_source: ?mesh_module.VertexSource = null,
};

// What a batch actually fetches binding zero from, which is what the recorder
// caches on. Keyed on the resolved source and not on the mesh: two batches of
// one mesh drawn from two prepass outputs would otherwise bind the first one's
// buffer and draw the same shape twice.
pub fn batchVertexSource(batch: RecordBatch) mesh_module.VertexSource {
    return batch.vertex_source orelse batch.mesh.baseSource();
}

pub const Renderer = struct {
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    allocator: Allocator,

    // Sized to the swapchain and replaced with it. One of each rather than one
    // per frame: the barriers at the top of the main pass order one frame's use
    // after the previous frame's.
    hdr: image.Image,
    depth: image.Image,

    scene_module: vk.ShaderModule,
    post_module: vk.ShaderModule,
    scene_layout: vk.PipelineLayout,
    post_layout: vk.PipelineLayout,
    scene_pipelines: [scene_pipeline_count]vk.Pipeline,
    post_pipeline: vk.Pipeline,

    frame: frame_set.FrameSet,
    // The scene's own set: one copy, holding the packed material array every
    // fragment indexes. The buffer behind it belongs to composition, so the set
    // exists from init and stays unusable until `setMaterialBuffer` fills it.
    scene: SceneSets,
    material_buffer_ready: bool,
    environment_ready: bool,
    materials: MaterialSets,
    // Descriptor sets and frame slots are indexed without checks after the
    // recorder validates the whole batch list. Material readiness changes at
    // cold setup; an instance count is published only after its frame update
    // succeeds.
    //
    // Null is a material whose textures have not been pointed at yet, so this
    // one array answers both what a batch draws through and whether it may be
    // drawn at all. The alpha mode belongs to the material rather than to the
    // batch: it does not vary between instances the way a cull mode does, which
    // a mirrored transform really can flip.
    material_modes: []?pipeline.Mode,
    frame_instance_counts: []usize,
    post_sets: post.Sets,
    post_sampler: vk.Sampler,

    // What the main pass clears its target to. Black hides the difference
    // between a pass that drew nothing and a post chain that carried nothing,
    // so a caller that wants to tell them apart sets it to something else.
    clear_colour: [4]f32 = .{ 0, 0, 0, 1 },

    pub fn init(
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        allocator: Allocator,
        extent: vk.Extent2D,
        frames: usize,
        capacity: frame_set.Capacity,
        material_capacity: u32,
        present_format: vk.Format,
        post_sampler: vk.Sampler,
    ) InitError!Renderer {
        var hdr = try attachment.createHdr(context, memory_allocator, extent);
        errdefer hdr.deinit();
        var depth = try attachment.createDepth(context, memory_allocator, extent);
        errdefer depth.deinit();

        const scene_module = try pipeline.createModule(context, shaders.scene.spirv);
        errdefer context.device.destroyShaderModule(scene_module, null);
        const post_module = try pipeline.createModule(context, shaders.fullscreen.spirv);
        errdefer context.device.destroyShaderModule(post_module, null);

        var frame = try frame_set.FrameSet.init(
            context,
            memory_allocator,
            allocator,
            frames,
            capacity,
        );
        errdefer frame.deinit(context, allocator);

        // One set for the whole scene. It is not sized by frame or material
        // count: the array it names covers every material at once and an
        // instance picks out of it by index.
        var scene = try SceneSets.init(context, allocator, 1);
        errdefer scene.deinit(context, allocator);

        // Set two has one long-lived descriptor set per material. It is not
        // multiplied by frame or batch count: frame data lives in set zero, and
        // every batch naming the same material reuses this set.
        var materials = try MaterialSets.init(context, allocator, material_capacity);
        errdefer materials.deinit(context, allocator);

        const material_modes = try allocator.alloc(?pipeline.Mode, material_capacity);
        errdefer allocator.free(material_modes);
        @memset(material_modes, null);

        const frame_instance_counts = try allocator.alloc(usize, frames);
        errdefer allocator.free(frame_instance_counts);
        @memset(frame_instance_counts, 0);

        var post_sets = try post.Sets.init(context, allocator, 1);
        errdefer post_sets.deinit(context, allocator);
        post.write(context, &post_sets, .{ .view = hdr.view, .sampler = post_sampler });

        // A pipeline layout names its sets by index, and the shader's `space` is
        // that index, so this array is in the order the constants above give.
        const scene_layout = try pipeline.createLayout(context, .{ .descriptor_sets = &.{
            frame.descriptorSetLayout(),
            scene.layout,
            materials.layout,
        } });
        errdefer context.device.destroyPipelineLayout(scene_layout, null);
        const post_layout = try pipeline.createLayout(context, .{
            .descriptor_sets = &.{post_sets.layout},
            .push_constants = &.{post.push_constant_range},
        });
        errdefer context.device.destroyPipelineLayout(post_layout, null);

        // One module, one layout and one set behind all four: they differ in the
        // vertex input and entry point the variant selects, and in the depth and
        // blend state the mode selects. That is what keeps both axes a pipeline
        // difference rather than a second descriptor layout.
        var scene_pipelines: [scene_pipeline_count]vk.Pipeline = undefined;
        var created: usize = 0;
        errdefer for (scene_pipelines[0..created]) |built|
            context.device.destroyPipeline(built, null);

        for (std.enums.values(SceneVariant)) |variant| {
            for (std.enums.values(pipeline.Mode)) |mode| {
                const skinned = variant == .skinned;
                // The rollback below walks the built prefix, so the loop has to
                // fill the array in index order.
                std.debug.assert(scenePipelineIndex(variant, mode) == created);
                scene_pipelines[created] = try pipeline.create(context, .{
                    .mode = mode,
                    .streams = .{ .skinned = skinned },
                    .culling = .dynamic,
                    .formats = .{ .colour = hdr.format, .depth = depth.format },
                    .layout = scene_layout,
                    .stages = .{
                        .vertex = .{
                            .module = scene_module,
                            .entry_point = if (skinned)
                                shaders.scene.skinned_vertex_entry.?
                            else
                                shaders.scene.vertex_entry.?,
                        },
                        .fragment = .{
                            .module = scene_module,
                            .entry_point = shaders.scene.fragment_entry.?,
                        },
                    },
                });
                created += 1;
            }
        }

        const post_pipeline = try pipeline.create(context, .{
            .mode = .solid,
            .streams = null,
            .culling = .{ .fixed = .{} },
            .formats = .{ .colour = present_format },
            .layout = post_layout,
            .stages = .{
                .vertex = .{ .module = post_module, .entry_point = shaders.fullscreen.vertex_entry.? },
                .fragment = .{ .module = post_module, .entry_point = shaders.fullscreen.fragment_entry.? },
            },
        });
        errdefer context.device.destroyPipeline(post_pipeline, null);

        return .{
            .context = context,
            .memory_allocator = memory_allocator,
            .allocator = allocator,
            .hdr = hdr,
            .depth = depth,
            .scene_module = scene_module,
            .post_module = post_module,
            .scene_layout = scene_layout,
            .post_layout = post_layout,
            .scene_pipelines = scene_pipelines,
            .post_pipeline = post_pipeline,
            .frame = frame,
            .scene = scene,
            .material_buffer_ready = false,
            .environment_ready = false,
            .materials = materials,
            .material_modes = material_modes,
            .frame_instance_counts = frame_instance_counts,
            .post_sets = post_sets,
            .post_sampler = post_sampler,
        };
    }

    // Vulkan specification, vkDestroyPipeline and the rest: every submission
    // naming any of these must have completed. The caller drains the device.
    pub fn deinit(self: *Renderer) void {
        const device = self.context.device;
        device.destroyPipeline(self.post_pipeline, null);
        for (self.scene_pipelines) |built| device.destroyPipeline(built, null);
        device.destroyPipelineLayout(self.post_layout, null);
        device.destroyPipelineLayout(self.scene_layout, null);
        self.post_sets.deinit(self.context, self.allocator);
        self.allocator.free(self.frame_instance_counts);
        self.allocator.free(self.material_modes);
        self.materials.deinit(self.context, self.allocator);
        self.scene.deinit(self.context, self.allocator);
        self.frame.deinit(self.context, self.allocator);
        device.destroyShaderModule(self.post_module, null);
        device.destroyShaderModule(self.scene_module, null);
        self.depth.deinit();
        self.hdr.deinit();
        self.* = undefined;
    }

    // Point the scene set at the buffer holding every packed material. Cold:
    // once, when composition has built the array. The descriptor covers the
    // whole allocation, so re-uploading materials into the same buffer does not
    // come back here; handing over a different buffer does, and the caller
    // ensures no submitted frame is reading the old one.
    pub fn setMaterialBuffer(self: *Renderer, storage: *const material_storage.MaterialStorage) void {
        material_storage.write(self.context, self.scene.set(0), storage);
        self.material_buffer_ready = true;
    }

    // Point the scene set at an environment. Cold, like the material buffer, and
    // for the same reason: it is scene state and not frame state.
    //
    // There is no neutral default written at init. The renderer does not own a
    // texture cache and so cannot produce the black cubemaps that stand in for
    // an absent environment; composition does, through `Environment.neutral`,
    // and passes them here. Recording refuses to proceed until it has.
    pub fn setEnvironment(self: *Renderer, source: environment.Environment) void {
        environment.write(self.context, self.scene.set(0), source);
        self.environment_ready = true;
    }

    // Point one material set at its textures and say how it composites. Cold:
    // once when that material's residency is established, not per frame or per
    // batch. The caller ensures no submitted frame is reading the set when
    // replacing an existing source.
    //
    // The alpha mode arrives as glTF states it and is mapped here, so the
    // pipeline's two-mode view of it is named in one place. Passing it with the
    // textures is what makes a material configured in one call: a set pointed at
    // its images but with no mode would be ready to bind and impossible to draw.
    pub fn setMaterialTextures(
        self: *Renderer,
        material_index: u32,
        textures: *const resource_storage.TextureSet,
        alpha_mode: res.MaterialInfo.Rendering.AlphaMode,
    ) MaterialError!void {
        try validateMaterialIndex(self.materials.sets.len, material_index);

        const infos = [_]vk.DescriptorImageInfo{
            .{
                .sampler = textures.base_colour.sampler,
                .image_view = textures.base_colour.view,
                .image_layout = pass.sampled_layout,
            },
            .{
                .sampler = textures.metallic_roughness.sampler,
                .image_view = textures.metallic_roughness.view,
                .image_layout = pass.sampled_layout,
            },
            .{
                .sampler = textures.normal.sampler,
                .image_view = textures.normal.view,
                .image_layout = pass.sampled_layout,
            },
            .{
                .sampler = textures.emissive.sampler,
                .image_view = textures.emissive.view,
                .image_layout = pass.sampled_layout,
            },
            .{
                .sampler = textures.occlusion.sampler,
                .image_view = textures.occlusion.view,
                .image_layout = pass.sampled_layout,
            },
        };
        var writes: [material_bindings.len]vk.WriteDescriptorSet = undefined;
        for (material_bindings, &infos, &writes) |binding, *info, *write| {
            write.* = .{
                .dst_set = self.materials.set(@intCast(material_index)),
                .dst_binding = binding.slot,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = binding.kind,
                .p_image_info = @ptrCast(info),
                .p_buffer_info = &no_buffers,
                .p_texel_buffer_view = &no_texel_buffers,
            };
        }
        self.context.device.updateDescriptorSets(&writes, null);
        self.material_modes[material_index] = pipeline.modeFor(alpha_mode);
    }

    // The attachments follow the swapchain, so a resize replaces them and the
    // post pass's descriptor with them.
    //
    // Candidate first: the old pair survives if either allocation fails, so a
    // resize that cannot be honoured leaves a renderer that still draws.
    pub fn resize(self: *Renderer, extent: vk.Extent2D) InitError!void {
        var hdr = try attachment.createHdr(self.context, self.memory_allocator, extent);
        errdefer hdr.deinit();
        var depth = try attachment.createDepth(self.context, self.memory_allocator, extent);
        errdefer depth.deinit();

        self.hdr.deinit();
        self.depth.deinit();
        self.hdr = hdr;
        self.depth = depth;
        post.write(self.context, &self.post_sets, .{
            .view = self.hdr.view,
            .sampler = self.post_sampler,
        });
    }

    pub fn targetExtent(self: *const Renderer) vk.Extent2D {
        return .{ .width = self.hdr.width, .height = self.hdr.height };
    }

    pub fn mainPassTarget(self: *const Renderer) pass.Target {
        return .{
            .hdr_image = self.hdr.handle,
            .hdr_view = self.hdr.view,
            .depth_image = self.depth.handle,
            .depth_view = self.depth.view,
            .extent = self.targetExtent(),
        };
    }

    // Fill the frame's slot. Separate from recording because the host writes it
    // after that slot's fence and before anything is recorded against it.
    //
    // `view_projection` is the scene's, with Y up. The flip that turns it into
    // what a framebuffer wants is applied here, which is the one place that
    // knows the target counts rows downward.
    //
    // The lights are in world space and the flip does not reach them: the
    // fragment shader lights the world position the vertex stage emitted, which
    // is before any clip transform.
    pub fn update(
        self: *Renderer,
        frame_index: usize,
        contents: frame_set.FrameSet.Frame,
    ) UpdateError!void {
        try validateFrameIndex(self.frame_instance_counts.len, frame_index);

        var flipped = contents;
        flipped.camera.view_projection = uniforms.vulkanClip(contents.camera.view_projection);
        try self.frame.update(frame_index, flipped);
        self.frame_instance_counts[frame_index] = contents.models.len;
    }

    fn scenePipelineFor(
        self: *const Renderer,
        streams: res.VertexStreams,
        mode: pipeline.Mode,
    ) vk.Pipeline {
        return self.scene_pipelines[scenePipelineIndex(sceneVariantFor(streams), mode)];
    }

    pub fn record(
        self: *const Renderer,
        command_buffer: vk.CommandBuffer,
        frame_index: usize,
        target: post.Target,
        batches: []const RecordBatch,
        post_settings: post.Settings,
    ) RecordError!void {
        const post_constants = try post.pushConstants(post_settings);

        if (batches.len > 0) {
            if (!self.material_buffer_ready) return error.MaterialBufferNotConfigured;
            if (!self.environment_ready) return error.EnvironmentNotConfigured;
            try validateFrameIndex(self.frame_instance_counts.len, frame_index);
            const live_instances = self.frame_instance_counts[frame_index];
            var blended_seen = false;
            for (batches) |batch| {
                try validateRecordBatch(
                    self.material_modes,
                    live_instances,
                    batch.material_index,
                    batch.cull_mode,
                    batch.first_instance,
                    batch.instance_count,
                    blended_seen,
                );
                // Validated above, so the mode is present and the index is in
                // range.
                if (self.material_modes[batch.material_index].? == .blended)
                    blended_seen = true;
            }
        }

        const device = self.context.device;
        pass.begin(self.context, command_buffer, self.mainPassTarget(), .{
            .clear_colour = self.clear_colour,
        });

        if (batches.len > 0) {
            self.frame.bind(self.context, command_buffer, self.scene_layout, frame_index);
            // Bound once beside the frame set. Nothing in the list below can
            // change which material array a fragment reads, only which index it
            // reads out of.
            device.cmdBindDescriptorSets(
                command_buffer,
                .graphics,
                self.scene_layout,
                scene_set_index,
                &.{self.scene.set(0)},
                &.{},
            );

            var last_pipeline: ?vk.Pipeline = null;
            var last_material: ?u32 = null;
            var last_cull_mode: ?u32 = null;
            var last_mesh: ?*const mesh_module.Mesh = null;
            var last_source: ?mesh_module.VertexSource = null;

            for (batches) |batch| {
                // The list was validated as a whole above, so every material it
                // names is configured.
                const selected_pipeline = self.scenePipelineFor(
                    batch.mesh.streams,
                    self.material_modes[batch.material_index].?,
                );
                if (last_pipeline == null or last_pipeline.? != selected_pipeline) {
                    device.cmdBindPipeline(command_buffer, .graphics, selected_pipeline);
                    last_pipeline = selected_pipeline;
                }

                const cull_mode = batch.cull_mode.toInt();
                if (last_cull_mode == null or last_cull_mode.? != cull_mode) {
                    device.cmdSetCullMode(command_buffer, batch.cull_mode);
                    last_cull_mode = cull_mode;
                }

                if (last_material == null or last_material.? != batch.material_index) {
                    device.cmdBindDescriptorSets(
                        command_buffer,
                        .graphics,
                        self.scene_layout,
                        material_set_index,
                        &.{self.materials.set(@intCast(batch.material_index))},
                        &.{},
                    );
                    last_material = batch.material_index;
                }

                const source = batchVertexSource(batch);
                if (last_mesh == null or
                    last_mesh.? != batch.mesh or
                    !std.meta.eql(last_source.?, source))
                {
                    batch.mesh.bind(self.context, command_buffer, source);
                    last_mesh = batch.mesh;
                    last_source = source;
                }

                drawInstanced(
                    self.context,
                    command_buffer,
                    batch.mesh,
                    batch.instance_count,
                    batch.first_instance,
                );
            }
        }
        pass.end(self.context, command_buffer, self.mainPassTarget());

        post.begin(self.context, command_buffer, target);
        device.cmdBindPipeline(command_buffer, .graphics, self.post_pipeline);
        device.cmdBindDescriptorSets(
            command_buffer,
            .graphics,
            self.post_layout,
            0,
            &.{self.post_sets.set(0)},
            &.{},
        );
        device.cmdPushConstants(
            command_buffer,
            self.post_layout,
            post.push_constant_range.stage_flags,
            post.push_constant_range.offset,
            post.push_constant_range.size,
            @ptrCast(&post_constants),
        );
        device.cmdDraw(command_buffer, post.vertex_count, 1, 0, 0);
        post.end(self.context, command_buffer, target);
    }
};

// `Mesh.draw` submits a single instance. Instancing is the renderer's, because
// how many copies of a mesh a frame holds is not a property of the mesh.
fn drawInstanced(
    context: *const Context,
    command_buffer: vk.CommandBuffer,
    mesh: *const mesh_module.Mesh,
    instances: u32,
    first_instance: u32,
) void {
    if (mesh.index_count > 0) {
        context.device.cmdDrawIndexed(command_buffer, mesh.index_count, instances, 0, 0, first_instance);
    } else {
        context.device.cmdDraw(command_buffer, mesh.vertex_count, instances, 0, first_instance);
    }
}

// Vulkan specification, VkWriteDescriptorSet: the members not selected by
// descriptorType are ignored, but the pointers are not optional.
const no_buffers = [_]vk.DescriptorBufferInfo{};
const no_texel_buffers = [_]vk.BufferView{};
