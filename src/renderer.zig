const std = @import("std");
const vk = @import("vulkan");
const attachment = @import("attachment.zig");
const Context = @import("context.zig").Context;
const descriptors = @import("descriptors.zig");
const frame_set = @import("frame_set.zig");
const image = @import("image.zig");
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

// What a material set holds. One texture for now, which is what the scene
// shader samples; the other four slots exist in a texture set and nothing reads
// them yet.
pub const material_bindings = [_]descriptors.Binding{
    .{ .slot = 0, .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
};

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
    EmptyBatch,
    InstanceRangeOutOfBounds,
    UnsupportedCullMode,
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
    material_ready: []const bool,
    live_instances: usize,
    material_index: u32,
    cull_mode: vk.CullModeFlags,
    first_instance: u32,
    instance_count: u32,
) RecordError!void {
    try validateMaterialIndex(material_ready.len, material_index);
    if (!material_ready[material_index]) return error.MaterialNotConfigured;
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
};

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
    scene_pipeline: vk.Pipeline,
    skinned_pipeline: vk.Pipeline,
    post_pipeline: vk.Pipeline,

    frame: frame_set.FrameSet,
    materials: MaterialSets,
    // Descriptor sets and frame slots are indexed without checks after the
    // recorder validates the whole batch list. Material readiness changes at
    // cold setup; an instance count is published only after its frame update
    // succeeds.
    material_ready: []bool,
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

        // Set one has one long-lived descriptor set per material. It is not
        // multiplied by frame or batch count: frame data lives in set zero, and
        // every batch naming the same material reuses this set.
        var materials = try MaterialSets.init(context, allocator, material_capacity);
        errdefer materials.deinit(context, allocator);

        const material_ready = try allocator.alloc(bool, material_capacity);
        errdefer allocator.free(material_ready);
        @memset(material_ready, false);

        const frame_instance_counts = try allocator.alloc(usize, frames);
        errdefer allocator.free(frame_instance_counts);
        @memset(frame_instance_counts, 0);

        var post_sets = try post.Sets.init(context, allocator, 1);
        errdefer post_sets.deinit(context, allocator);
        post.write(context, &post_sets, .{ .view = hdr.view, .sampler = post_sampler });

        // Set 0 is the frame's and set 1 the material's, in that order: a
        // pipeline layout names its sets by index, and the shader's `space` is
        // that index.
        const scene_layout = try pipeline.createLayout(context, .{ .descriptor_sets = &.{
            frame.descriptorSetLayout(),
            materials.layout,
        } });
        errdefer context.device.destroyPipelineLayout(scene_layout, null);
        const post_layout = try pipeline.createLayout(context, .{
            .descriptor_sets = &.{post_sets.layout},
            .push_constants = &.{post.push_constant_range},
        });
        errdefer context.device.destroyPipelineLayout(post_layout, null);

        const scene_pipeline = try pipeline.create(context, .{
            .mode = .solid,
            .streams = .{},
            .culling = .dynamic,
            .formats = .{ .colour = hdr.format, .depth = depth.format },
            .layout = scene_layout,
            .stages = .{
                .vertex = .{ .module = scene_module, .entry_point = shaders.scene.vertex_entry },
                .fragment = .{ .module = scene_module, .entry_point = shaders.scene.fragment_entry },
            },
        });
        errdefer context.device.destroyPipeline(scene_pipeline, null);

        // The same module, the same layout and the same set: the two differ in
        // the vertex input and the entry point taken from the module, which is
        // what keeps skinning a pipeline difference rather than a second
        // descriptor layout.
        //
        // Two named pipelines rather than a table keyed by `streams.index()`.
        // The colour and second-UV streams have no path in the shader yet, so a
        // keyed table would build six pipelines that differ from these two in
        // nothing a draw can reach.
        const skinned_pipeline = try pipeline.create(context, .{
            .mode = .solid,
            .streams = .{ .skinned = true },
            .culling = .dynamic,
            .formats = .{ .colour = hdr.format, .depth = depth.format },
            .layout = scene_layout,
            .stages = .{
                .vertex = .{ .module = scene_module, .entry_point = shaders.scene.skinned_vertex_entry },
                .fragment = .{ .module = scene_module, .entry_point = shaders.scene.fragment_entry },
            },
        });
        errdefer context.device.destroyPipeline(skinned_pipeline, null);

        const post_pipeline = try pipeline.create(context, .{
            .mode = .solid,
            .streams = null,
            .culling = .{ .fixed = .{} },
            .formats = .{ .colour = present_format },
            .layout = post_layout,
            .stages = .{
                .vertex = .{ .module = post_module, .entry_point = shaders.fullscreen.vertex_entry },
                .fragment = .{ .module = post_module, .entry_point = shaders.fullscreen.fragment_entry },
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
            .scene_pipeline = scene_pipeline,
            .skinned_pipeline = skinned_pipeline,
            .post_pipeline = post_pipeline,
            .frame = frame,
            .materials = materials,
            .material_ready = material_ready,
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
        device.destroyPipeline(self.skinned_pipeline, null);
        device.destroyPipeline(self.scene_pipeline, null);
        device.destroyPipelineLayout(self.post_layout, null);
        device.destroyPipelineLayout(self.scene_layout, null);
        self.post_sets.deinit(self.context, self.allocator);
        self.allocator.free(self.frame_instance_counts);
        self.allocator.free(self.material_ready);
        self.materials.deinit(self.context, self.allocator);
        self.frame.deinit(self.context, self.allocator);
        device.destroyShaderModule(self.post_module, null);
        device.destroyShaderModule(self.scene_module, null);
        self.depth.deinit();
        self.hdr.deinit();
        self.* = undefined;
    }

    // Point one material set at its textures. Cold: once when that material's
    // residency is established, not per frame or per batch. The caller ensures
    // no submitted frame is reading the set when replacing an existing source.
    pub fn setMaterialTextures(
        self: *Renderer,
        material_index: u32,
        textures: *const resource_storage.TextureSet,
    ) MaterialError!void {
        try validateMaterialIndex(self.materials.sets.len, material_index);

        const info = vk.DescriptorImageInfo{
            .sampler = textures.base_colour.sampler,
            .image_view = textures.base_colour.view,
            .image_layout = pass.sampled_layout,
        };
        const writes = [_]vk.WriteDescriptorSet{.{
            .dst_set = self.materials.set(@intCast(material_index)),
            .dst_binding = material_bindings[0].slot,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = material_bindings[0].kind,
            .p_image_info = @ptrCast(&info),
            .p_buffer_info = &no_buffers,
            .p_texel_buffer_view = &no_texel_buffers,
        }};
        self.context.device.updateDescriptorSets(&writes, null);
        self.material_ready[material_index] = true;
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

    fn scenePipelineFor(self: *const Renderer, streams: res.VertexStreams) vk.Pipeline {
        return switch (sceneVariantFor(streams)) {
            .unskinned => self.scene_pipeline,
            .skinned => self.skinned_pipeline,
        };
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
            try validateFrameIndex(self.frame_instance_counts.len, frame_index);
            const live_instances = self.frame_instance_counts[frame_index];
            for (batches) |batch| {
                try validateRecordBatch(
                    self.material_ready,
                    live_instances,
                    batch.material_index,
                    batch.cull_mode,
                    batch.first_instance,
                    batch.instance_count,
                );
            }
        }

        const device = self.context.device;
        pass.begin(self.context, command_buffer, self.mainPassTarget(), .{
            .clear_colour = self.clear_colour,
        });

        if (batches.len > 0) {
            self.frame.bind(self.context, command_buffer, self.scene_layout, frame_index);

            var last_pipeline: ?vk.Pipeline = null;
            var last_material: ?u32 = null;
            var last_cull_mode: ?u32 = null;
            var last_mesh: ?*const mesh_module.Mesh = null;

            for (batches) |batch| {
                const selected_pipeline = self.scenePipelineFor(batch.mesh.streams);
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
                        1,
                        &.{self.materials.set(@intCast(batch.material_index))},
                        &.{},
                    );
                    last_material = batch.material_index;
                }

                if (last_mesh == null or last_mesh.? != batch.mesh) {
                    batch.mesh.bind(self.context, command_buffer);
                    last_mesh = batch.mesh;
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
