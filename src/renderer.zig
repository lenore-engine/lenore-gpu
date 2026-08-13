const std = @import("std");
const vk = @import("vulkan");
const attachment = @import("pass/attachment.zig");
const bloom = @import("pass/bloom.zig");
const Context = @import("device/context.zig").Context;
const descriptors = @import("binding/descriptors.zig");
const environment = @import("object/environment.zig");
const frame_set = @import("binding/frame_set.zig");
const image = @import("object/image.zig");
const materials_module = @import("binding/materials.zig");
const memory = @import("memory/allocator.zig");
const mesh_module = @import("object/mesh.zig");
const pass = @import("pass/scene.zig");
const pipeline = @import("binding/pipeline.zig");
const post = @import("pass/post.zig");
const res = @import("lenore-resources");
const resource_storage = @import("store/resources.zig");
const shadow = @import("pass/shadow.zig");
const sky = @import("pass/sky.zig");
const uniforms = @import("binding/uniforms.zig");

const Allocator = std.mem.Allocator;

const log = std.log.scoped(.vulkan);

// The main pass and the pass that presents its result, and everything they
// need that outlives a frame.
//
// What it does not own: meshes, textures and materials, which belong to the
// storage the engine fills, and the swapchain, which the frame loop drives. It
// is handed the image to present into.

// Which set each layout occupies, in the order the pipeline layout names them,
// which is also the space a shader's binding declares. They are in increasing
// order of how often the recorder binds them: the frame set once, the scene set
// once, a material set whenever the ordered list reaches a new material.
pub const frame_set_index = 0;
pub const scene_set_index = 1;
pub const material_set_index = 2;
// Last, and outside the scene set on purpose: the bake's own pipeline layout
// names the three before it, so a set in that range would be bound while the map
// it holds is the attachment being written. See `shadow.bindings`.
pub const shadow_set_index = 3;

// The whole scene set, assembled from the two files that write into it: the
// packed material array every fragment indexes, then the environment every
// fragment lights itself from. Neither list belongs in the other's file, and
// joining them here is what makes a slot claimed twice a compile error rather
// than a descriptor quietly overwritten at run time.
pub const scene_bindings = materials_module.bindings ++ environment.bindings;

const SceneSets = descriptors.Sets(&scene_bindings);

// What a material set holds: base colour, metallic-roughness, normal, emissive,
// then occlusion. One binding per slot of `resource_storage.TextureSet`, in that
// type's field order, which is also the order `materials_module.TextureSlot`
// indexes the packed transform array by.
pub const material_bindings = [_]descriptors.Binding{
    .{ .slot = 0, .name = "base_colour", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
    .{ .slot = 1, .name = "metallic_roughness", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
    .{ .slot = 2, .name = "normal", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
    .{ .slot = 3, .name = "emissive", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
    .{ .slot = 4, .name = "occlusion", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
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
    shadow.InitError ||
    bloom.InitError ||
    pipeline.CreateError;

// What the renderer keeps about a material once it has been pointed at its
// textures: everything a draw of it needs that the descriptor set does not
// carry.
//
// The authored alpha mode is stored rather than the blend mode it maps to. The
// two passes need different things out of it: the main pass wants to know
// whether to blend, and MASK and OPAQUE answer the same there, while the bake
// wants to know whether to run a cutoff test, where they do not. Keeping the
// glTF value means neither question is answered from the other's collapse of it.
pub const MaterialRecord = struct {
    alpha: res.MaterialInfo.Rendering.AlphaMode,
    // The cutoff test the bake pushes for a masked caster, in the units the
    // shader compares in.
    mask: shadow.PushConstants,

    pub fn mode(self: MaterialRecord) pipeline.Mode {
        return pipeline.modeFor(self.alpha);
    }

    // Built from the authored material in one place, so the cutoff and the
    // factor it is compared against cannot come from two different materials.
    pub fn forMaterial(info: *const res.MaterialInfo) MaterialRecord {
        return .{
            .alpha = info.rendering.alpha_mode,
            .mask = .{
                .alpha_cutoff = info.rendering.alpha_cutoff,
                // glTF 2.0 section 3.9.2: the factor multiplies the texture,
                // alpha included, so the test the bake runs is against the
                // product and not against the sampled value alone.
                .factor_alpha = info.factors.base_colour[3],
            },
        };
    }
};

// Whether this frame re-records the sun shadow map.
//
// The decision is the caller's because the renderer cannot make it: what the map
// has to hold changes when the sun moves or when a caster does, and neither is
// something a renderer is told about. A scene that neither animates nor moves
// its sun bakes once and reuses that map for every frame after.
pub const ShadowBake = enum { reuse, rebake };

pub const MaterialError = error{MaterialIndexOutOfRange};
pub const FrameError = error{FrameIndexOutOfRange};
pub const UpdateError = frame_set.UpdateError || FrameError;
pub const RecordError = MaterialError || FrameError || post.SettingsError ||
    bloom.SettingsError || error{
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
    // The visible count is past the end of the list it counts into. It slices
    // that list, and the two arrive as separate members of one request with
    // nothing in the types to say they were produced together.
    VisibleCountOutOfRange,
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
    material_records: []const ?MaterialRecord,
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
    try validateMaterialIndex(material_records.len, material_index);
    const material = material_records[material_index] orelse return error.MaterialNotConfigured;
    const mode = material.mode();
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

// Where in a batch list the background is recorded: before the batch at this
// index, or after all of them when it equals the list's length. Null is a
// recording that draws none.
//
// The background belongs between the two layers the list is partitioned into.
// After the solid batches, because the depth test against what they wrote is
// the only thing keeping it behind them, and before the blended ones, because
// those composite over whatever is in the target and the background is what
// should be under them.
//
// Split from the recording for the reason `validateRecordBatch` is: the decision
// is a function of the list alone, and a device is what stands between it and a
// test. It takes the first blended index rather than finding one, because the
// walk that validates the list has already answered that question and two
// answers to it could differ.
pub fn backgroundSlot(first_blended: ?usize, batch_count: usize, background: sky.Background) ?usize {
    return switch (background) {
        .clear => null,
        .environment => first_blended orelse batch_count,
    };
}

// One frame as composition asks for it, before anything about it is checked.
//
// None of it is retained, for the reason `post.Settings` is not: what has
// already been submitted must not change when a later frame asks for something
// else.
pub const RecordRequest = struct {
    frame_index: usize,
    // Every batch this frame, with the ones the camera can see first.
    batches: []const RecordBatch,
    // How many of them that is. The main pass draws the prefix and the shadow
    // bake draws all of them, because a caster the camera cannot see still casts
    // into what it can and an orthographic sun fit around the whole scene holds
    // every one of them.
    visible: usize,
    background: sky.Background,
    // Null is a frame that runs no chain and composites none. Not a look with
    // everything at zero: that would still cost the chain, and the composite
    // would still sample it.
    bloom: ?bloom.Settings,
    post: post.Settings,
};

// What the renderer's own state contributes to a plan.
//
// Named apart from the renderer so that planning is reachable from a test, for
// the reason `validateRecordBatch` takes slices rather than a renderer: every
// member here is a slice, a bool or a count, and none of it needs a device.
pub const RecordState = struct {
    material_records: []const ?MaterialRecord,
    frame_instance_counts: []const usize,
    material_buffer_ready: bool,
    environment_ready: bool,
    // The depth the look is resolved against. The composite's weight is the
    // reciprocal of a series whose length is the chain's depth, so a look is
    // only meaningful beside the chain that will carry it.
    bloom_levels: u32,
};

// A frame that has been validated and resolved, and the only thing `record`
// accepts.
//
// Producing one is where every fallible step of a frame happens. Recording can
// then not fail, so no error path exists that could leave a rendering instance
// open or a frame half recorded. That was already true when both halves were one
// function, by every `try` in it having been written above `pass.begin` and
// staying there; here it holds however the recording half is edited.
//
// It carries both batch lists rather than `record` taking one of its own. Each
// is what the checks were made against, and a second parameter would let a
// recording walk one they never saw.
//
// Every member is something a check established or a resolution produced.
// Whether the shadow map is rebaked is neither, so it stays an argument to
// `record` beside the target: a field here that planning never looks at would
// leave the type promising more than it holds.
pub const RecordPlan = struct {
    frame_index: usize,
    // What the main pass draws: the prefix the camera can see.
    batches: []const RecordBatch,
    // What the shadow bake draws: every batch, culled ones included. It is a
    // superset of `batches` and begins with it.
    casters: []const RecordBatch,
    // Where the background is recorded, or null for a recording that draws none.
    background_slot: ?usize,
    look: ?bloom.Look,
    post_constants: post.PushConstants,
};

// Checks a request against the state that will record it, and resolves what the
// recording reads instead of the settings it was asked for.
pub fn planRecording(state: RecordState, request: RecordRequest) RecordError!RecordPlan {
    const look: ?bloom.Look = if (request.bloom) |settings|
        try bloom.resolve(settings, state.bloom_levels)
    else
        null;
    const post_constants = try post.pushConstants(request.post, look);
    if (request.visible > request.batches.len) return error.VisibleCountOutOfRange;

    // Both read the frame set's camera and the scene set's environment, so a
    // background over an empty list has the same preconditions a batch does,
    // minus the material array it does not index. Keyed on the whole list rather
    // than the visible part: the bake binds the same frame set, so a frame with
    // nothing on screen and casters behind the camera still needs both.
    if (request.batches.len > 0 or request.background == .environment) {
        if (!state.environment_ready) return error.EnvironmentNotConfigured;
        try validateFrameIndex(state.frame_instance_counts.len, request.frame_index);
    }

    // Where the layers meet, and what the ordering rule is checked against: a
    // batch validates only if no blended one came before it, so this is both the
    // boundary the background is recorded at and the state that answers the rule.
    var first_blended: ?usize = null;
    if (request.batches.len > 0) {
        if (!state.material_buffer_ready) return error.MaterialBufferNotConfigured;
        const live_instances = state.frame_instance_counts[request.frame_index];
        for (request.batches, 0..) |batch, index| {
            const in_view = index < request.visible;
            try validateRecordBatch(
                state.material_records,
                live_instances,
                batch.material_index,
                batch.cull_mode,
                batch.first_instance,
                batch.instance_count,
                // The ordering rule is the compositing rule, and only the pass
                // that composites has one. Past the boundary the bake writes
                // depth and blends nothing, so a solid batch after a blended one
                // is not a plan that cannot be drawn there.
                in_view and first_blended != null,
            );
            // Validated above, so the record is present and the index is in
            // range.
            if (in_view and first_blended == null and
                state.material_records[batch.material_index].?.mode() == .blended)
                first_blended = index;
        }
    }

    return .{
        .frame_index = request.frame_index,
        .batches = request.batches[0..request.visible],
        .casters = request.batches,
        // Against the visible count, because the background sits between the
        // layers of the pass that draws it and that pass ends at the boundary.
        .background_slot = backgroundSlot(first_blended, request.visible, request.background),
        .look = look,
        .post_constants = post_constants,
    };
}

// Which of the scene pipelines a mesh draws through: the optional vertex
// streams that change what the shader does, and nothing else.
pub const SceneVariant = struct {
    skinned: bool,
    uv1: bool,
    colour: bool,

    // The vertex input a pipeline of this variant declares. Read by both the
    // input and the entry point at creation, so the two cannot name different
    // sets of bindings: a shader reading location 7 with no binding 3 behind it
    // is not a pipeline that can be created.
    pub fn streams(self: SceneVariant) res.VertexStreams {
        return .{ .skinned = self.skinned, .uv1 = self.uv1, .colour = self.colour };
    }
};

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
pub fn sceneVariantFor(streams: res.VertexStreams) SceneVariant {
    return .{ .skinned = streams.skinned, .uv1 = streams.uv1, .colour = streams.colour };
}

// The scene pipelines, one per vertex variant and blend mode. Both axes change
// what a draw does: the variant picks the vertex entry point and its input, and
// the mode decides whether the draw blends and whether it writes depth.
//
// Eight: four vertex variants against two blend modes. The key is not
// `streams.index()`, which would put the colour stream on an axis of its own and
// build pipelines differing in nothing a draw can reach.
//
// The modes are named rather than counted off `pipeline.Mode`. That enum also
// carries the background, which is one pipeline of its own and not a point on
// this axis, so a count taken from the enum would build a row of scene
// pipelines that nothing can select and that no draw would ever bind.
pub const scene_modes = [_]pipeline.Mode{ .solid, .blended };
pub const scene_variants = 8;
pub const scene_pipeline_count = scene_variants * scene_modes.len;

comptime {
    // What the index arithmetic below rests on: the modes above are the enum's
    // first members, in the enum's own order. Reordering `pipeline.Mode` without
    // this would keep every index in range and hand half the draws the other
    // mode's pipeline.
    for (scene_modes, 0..) |mode, index| std.debug.assert(@intFromEnum(mode) == index);
}

// Where a variant sits on the vertex axis alone, which is the order a supplied
// shader lists its vertex entry points in.
//
// One account of the pair, read by the entry point a pipeline is built with and
// by the slot it is built into. Two would be a shader set indexed differently
// from the pipelines it feeds, and every draw would then be transformed by the
// wrong variant's vertex stage while nothing said so.
pub fn sceneVariantIndex(variant: SceneVariant) usize {
    return @as(usize, @intFromBool(variant.skinned)) * 4 +
        @as(usize, @intFromBool(variant.uv1)) * 2 +
        @intFromBool(variant.colour);
}

// Public so that the one property the creation loop depends on can be checked
// without a device: that this is a bijection onto the array's indices. The loop
// fills the array in index order and the rollback walks the built prefix, so a
// collision or a value past the end is an uninitialised pipeline or a write off
// the end of the array.
pub fn scenePipelineIndex(variant: SceneVariant, mode: pipeline.Mode) usize {
    // Justified by construction rather than by the check: every mode reaching
    // here comes from `modeFor`, which maps the three glTF alpha modes onto the
    // two above and cannot name the background.
    std.debug.assert(@intFromEnum(mode) < scene_modes.len);

    return sceneVariantIndex(variant) * scene_modes.len + @intFromEnum(mode);
}

// What a main-pass shader has to supply for the sixteen scene pipelines to be
// built from it.
//
// The vertex entry points are an array and not eight named fields, because the
// axes they run over are the ones `sceneVariantIndex` already computes: the
// optional skin stream, the second UV set and the vertex colour, in that order
// of significance. A named field per case would be a second account of that
// triple, and the two would have to be kept in step by reading both.
//
// Two fragment entry points, and the axis is the colour stream alone. A
// fragment stage's input has to be the vertex stage's output, so a variant
// carrying COLOR_0 cannot be shaded by an entry point that does not declare it.
// The blend mode is not an axis here: it is a pipeline state rather than a
// shading difference.
pub const SceneShader = struct {
    spirv: []const u32,
    vertex_entry: [scene_variants][*:0]const u8,
    fragment_entry: [*:0]const u8,
    // For the variants whose vertex stage carries COLOR_0.
    colour_fragment_entry: [*:0]const u8,
};

// Every shader the renderer builds a pipeline from. Grouped rather than passed
// one per parameter: five values of the same shape in an argument list is five
// chances to swap two of them, and none of the swaps is a type error.
pub const Shaders = struct {
    scene: SceneShader,
    sky: sky.Shader,
    post: post.Shader,
    bloom: bloom.Shader,
    shadow: shadow.Shader,
};

// The Vulkan-facing form the umbrella translates a scene batch into. Keeping
// this distinct from the scene plan is deliberate: the GPU module owns resource
// pointers and Vulkan cull state, while the scene module owns ordering policy.
// Composition performs that translation into preallocated storage explicitly.
pub const RecordBatch = struct {
    mesh: *const mesh_module.Mesh,
    material_index: u32,
    cull_mode: vk.CullModeFlags,
    // Which winding this batch's rasterizer calls the front. A mirrored draw
    // arrives with the opposite one, and it is set even where nothing is
    // culled: `SV_IsFrontFace` comes from the same state, and a double-sided
    // material reads it to decide which way its normals point.
    front_face: vk.FrontFace,
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
    sky_module: vk.ShaderModule,
    post_module: vk.ShaderModule,
    scene_layout: vk.PipelineLayout,
    post_layout: vk.PipelineLayout,
    scene_pipelines: [scene_pipeline_count]vk.Pipeline,
    // Built against `scene_layout`, which is what lets it be drawn between two
    // scene batches without rebinding anything.
    sky_pipeline: vk.Pipeline,
    post_pipeline: vk.Pipeline,
    // The post pass with the bloom chain composited in. A second pipeline and
    // not a weight of zero: the entry point behind this one samples the chain
    // and the other never names it, so a recording that did not run the chain
    // cannot read what its memory happened to hold.
    post_bloom_pipeline: vk.Pipeline,

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
    material_records: []?MaterialRecord,
    frame_instance_counts: []usize,

    shadows: shadow.ShadowPass,
    // Whether the map holds anything a lookup may read. False until the first
    // pass over it, which is why `record` opens and closes the bake even when it
    // was not asked to: a cleared map reads as fully lit, and an untouched one
    // reads as whatever the memory held.
    shadow_valid: bool,
    // How many frames have re-recorded the casters. What a caller asked for and
    // what was recorded are two different things, and this is the only one of
    // them a caller can read back: a scene that has stopped moving under a still
    // sun should stop adding to it.
    shadow_bakes: u64,
    // How many recordings have drawn a background. The counterpart of
    // `shadow_bakes` and read for the same reason: what a caller asked for and
    // what was recorded are two different things, and this is the one a caller
    // can check from outside a frame it cannot see into.
    background_draws: u64,

    // The chain, built whatever the look asks for. It follows the target's size
    // and not the setting, because a pass created on the first frame that wants
    // it would allocate an image inside a frame. At half the target on each axis
    // and a third of that again for the levels under it, it is a fraction of
    // what the target already costs.
    bloom: bloom.BloomPass,
    // How many recordings have run the chain. The counterpart of `shadow_bakes`
    // and `background_draws`, and read for the same reason: what a caller asked
    // for and what was recorded are two different things, and this is the one a
    // caller can check from outside a frame it cannot see into.
    bloom_chains: u64,

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
        // Square, and the number the scene's `SunShadowFit` has to be computed
        // against. Asked for here rather than fixed, because what a map costs
        // and what it resolves are the caller's trade to make.
        shadow_map_size: u32,
        shaders: Shaders,
    ) InitError!Renderer {
        var hdr = try attachment.createHdr(context, memory_allocator, extent);
        errdefer hdr.deinit();
        var depth = try attachment.createDepth(context, memory_allocator, extent);
        errdefer depth.deinit();

        // Both are the first candidate the device's features carried, so which
        // one it is belongs in the log beside the present mode. A resize takes
        // the same lists to the same device and cannot change them, so this is
        // reported here and not in `resize`.
        log.info("attachments: hdr {t}, depth {t}", .{ hdr.format, depth.format });

        const scene_module = try pipeline.createModule(context, shaders.scene.spirv);
        errdefer context.device.destroyShaderModule(scene_module, null);
        const sky_module = try pipeline.createModule(context, shaders.sky.spirv);
        errdefer context.device.destroyShaderModule(sky_module, null);
        const post_module = try pipeline.createModule(context, shaders.post.spirv);
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

        const material_records = try allocator.alloc(?MaterialRecord, material_capacity);
        errdefer allocator.free(material_records);
        @memset(material_records, null);

        const frame_instance_counts = try allocator.alloc(usize, frames);
        errdefer allocator.free(frame_instance_counts);
        @memset(frame_instance_counts, 0);

        var post_sets = try post.Sets.init(context, allocator, 1);
        errdefer post_sets.deinit(context, allocator);

        // Before the post set is written, because that set names the chain's
        // finest level and the chain is where that view comes from.
        var bloom_pass = try bloom.BloomPass.init(
            context,
            memory_allocator,
            allocator,
            extent,
            hdr.format,
            hdr.view,
            shaders.bloom,
        );
        errdefer bloom_pass.deinit();

        post.write(context, &post_sets, .{
            .view = hdr.view,
            .sampler = post_sampler,
            .bloom_view = bloom_pass.compositeView(),
            .bloom_sampler = bloom_pass.compositeSampler(),
        });

        // Before the scene layout, which names the set it owns. The three
        // layouts it is given are the first three of that same list, which is
        // what makes the bake's layout a prefix of the main pass's rather than a
        // second arrangement of the same sets.
        var shadows = try shadow.ShadowPass.init(
            context,
            memory_allocator,
            allocator,
            shadow_map_size,
            .{
                .frame = frame.descriptorSetLayout(),
                .scene = scene.layout,
                .material = materials.layout,
            },
            shaders.shadow,
        );
        errdefer shadows.deinit();

        // A pipeline layout names its sets by index, and the shader's `space` is
        // that index, so this array is in the order the constants above give.
        const scene_layout = try pipeline.createLayout(context, .{ .descriptor_sets = &.{
            frame.descriptorSetLayout(),
            scene.layout,
            materials.layout,
            shadows.descriptorSetLayout(),
        } });
        errdefer context.device.destroyPipelineLayout(scene_layout, null);
        const post_layout = try pipeline.createLayout(context, .{
            .descriptor_sets = &.{post_sets.layout},
            .push_constants = &.{post.push_constant_range},
        });
        errdefer context.device.destroyPipelineLayout(post_layout, null);

        // One module, one layout and one set behind all eight: they differ in
        // the vertex input and entry point the variant selects, and in the depth
        // and blend state the mode selects. That is what keeps both axes a
        // pipeline difference rather than a second descriptor layout.
        var scene_pipelines: [scene_pipeline_count]vk.Pipeline = undefined;
        var created: usize = 0;
        errdefer for (scene_pipelines[0..created]) |built|
            context.device.destroyPipeline(built, null);

        for ([_]bool{ false, true }) |skinned| {
            for ([_]bool{ false, true }) |uv1| {
                for ([_]bool{ false, true }) |colour| {
                    const variant: SceneVariant = .{
                        .skinned = skinned,
                        .uv1 = uv1,
                        .colour = colour,
                    };
                    for (scene_modes) |mode| {
                        // The rollback below walks the built prefix, so the
                        // loop has to fill the array in index order.
                        std.debug.assert(scenePipelineIndex(variant, mode) == created);
                        scene_pipelines[created] = try pipeline.create(context, .{
                            .mode = mode,
                            .streams = variant.streams(),
                            .culling = .dynamic,
                            .formats = .{ .colour = hdr.format, .depth = depth.format },
                            .layout = scene_layout,
                            .stages = .{
                                .vertex = .{
                                    .module = scene_module,
                                    .entry_point = shaders.scene.vertex_entry[sceneVariantIndex(variant)],
                                },
                                // The fragment stage's input is the vertex
                                // stage's output, so the colour axis selects
                                // here as well.
                                .fragment = .{
                                    .module = scene_module,
                                    .entry_point = if (colour)
                                        shaders.scene.colour_fragment_entry
                                    else
                                        shaders.scene.fragment_entry,
                                },
                            },
                        });
                        created += 1;
                    }
                }
            }
        }

        // What the driver made of each scene pipeline, when the build asked to
        // be able to see it. Reported here rather than exposed through an
        // accessor: the handles belong to this renderer and outliving it is the
        // one way to read them wrong.
        if (pipeline.capture_statistics) {
            if (context.shader_statistics_enabled) {
                for (scene_pipelines[0..created], 0..) |built, index|
                    reportShaderStatistics(context, allocator, built, index);
            } else {
                log.warn(
                    "shader statistics were built for but the device has no " ++
                        "VK_KHR_pipeline_executable_properties",
                    .{},
                );
            }
        }

        // The same layout the scene pipelines were built with, so the frame and
        // scene sets bound for the draws on either side of it serve this one as
        // they stand. It reads neither of the two sets after them.
        const sky_pipeline = try pipeline.create(context, sky.config(
            scene_layout,
            sky_module,
            shaders.sky,
            .{ .colour = hdr.format, .depth = depth.format },
        ));
        errdefer context.device.destroyPipeline(sky_pipeline, null);

        // Two, differing in the fragment entry point alone. One layout and one
        // set serve both, so which is bound is the only thing a recording
        // decides about the composite.
        const post_pipeline = try pipeline.create(context, postConfig(
            post_layout,
            post_module,
            shaders.post,
            present_format,
            shaders.post.fragment_entry,
        ));
        errdefer context.device.destroyPipeline(post_pipeline, null);

        const post_bloom_pipeline = try pipeline.create(context, postConfig(
            post_layout,
            post_module,
            shaders.post,
            present_format,
            shaders.post.bloom_fragment_entry,
        ));
        errdefer context.device.destroyPipeline(post_bloom_pipeline, null);

        return .{
            .context = context,
            .memory_allocator = memory_allocator,
            .allocator = allocator,
            .hdr = hdr,
            .depth = depth,
            .scene_module = scene_module,
            .sky_module = sky_module,
            .post_module = post_module,
            .scene_layout = scene_layout,
            .post_layout = post_layout,
            .scene_pipelines = scene_pipelines,
            .sky_pipeline = sky_pipeline,
            .post_pipeline = post_pipeline,
            .post_bloom_pipeline = post_bloom_pipeline,
            .frame = frame,
            .scene = scene,
            .material_buffer_ready = false,
            .environment_ready = false,
            .materials = materials,
            .material_records = material_records,
            .frame_instance_counts = frame_instance_counts,
            .shadows = shadows,
            .shadow_valid = false,
            .shadow_bakes = 0,
            .background_draws = 0,
            .bloom = bloom_pass,
            .bloom_chains = 0,
            .post_sets = post_sets,
            .post_sampler = post_sampler,
        };
    }

    // Vulkan specification, vkDestroyPipeline and the rest: every submission
    // naming any of these must have completed. The caller drains the device.
    pub fn deinit(self: *Renderer) void {
        const device = self.context.device;
        device.destroyPipeline(self.post_bloom_pipeline, null);
        device.destroyPipeline(self.post_pipeline, null);
        device.destroyPipeline(self.sky_pipeline, null);
        for (self.scene_pipelines) |built| device.destroyPipeline(built, null);
        device.destroyPipelineLayout(self.post_layout, null);
        device.destroyPipelineLayout(self.scene_layout, null);
        self.post_sets.deinit(self.context, self.allocator);
        self.bloom.deinit();
        self.shadows.deinit();
        self.allocator.free(self.frame_instance_counts);
        self.allocator.free(self.material_records);
        self.materials.deinit(self.context, self.allocator);
        self.scene.deinit(self.context, self.allocator);
        self.frame.deinit(self.context, self.allocator);
        device.destroyShaderModule(self.post_module, null);
        device.destroyShaderModule(self.sky_module, null);
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
    pub fn setMaterialBuffer(self: *Renderer, storage: *const materials_module.MaterialStorage) void {
        materials_module.write(self.context, self.scene.set(0), storage);
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

    // Forget every material this renderer was told about, so a different scene
    // can configure the same sets.
    //
    // Only the records are cleared. The descriptor sets keep whatever images
    // they were pointed at, which after a scene is released are destroyed, and
    // that is exactly why the records go: `plan` refuses a batch whose material
    // has no record, so a stale index is rejected instead of binding a set that
    // names an image nobody owns. Leaving the records in place would keep the
    // refusal green while the guarantee behind it was gone.
    //
    // The caller drains the device first, for the reason `setMaterialTextures`
    // states: no submitted frame may be reading a set that is about to be
    // rewritten.
    pub fn clearMaterials(self: *Renderer) void {
        @memset(self.material_records, null);
        self.material_buffer_ready = false;
    }

    // Point one material set at its textures and say how it composites. Cold:
    // once when that material's residency is established, not per frame or per
    // batch. The caller ensures no submitted frame is reading the set when
    // replacing an existing source.
    //
    // The record arrives as the authored material states it and is mapped where
    // each pass needs it, so glTF's three alpha modes are collapsed to two in
    // one place and only for the pass that has two. Passing it with the textures
    // is what makes a material configured in one call: a set pointed at its
    // images but with no record would be ready to bind and impossible to draw.
    pub fn setMaterialTextures(
        self: *Renderer,
        material_index: u32,
        textures: *const resource_storage.TextureSet,
        material: MaterialRecord,
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
        self.material_records[material_index] = material;
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

        // The chain follows the target, so a resize rebuilds it, and its depth
        // can change with it: a smaller window supports fewer levels. Whatever
        // it comes back with is what the next `resolve` normalizes against,
        // which is why the composite's weight is derived per recording rather
        // than held beside the settings.
        //
        // Before the two above are installed, and against the candidate's view
        // rather than the field's. The errdefers here name the candidates, so
        // anything failing after they were installed would destroy the images
        // the renderer had just started using; `BloomPass.recreate` keeps its
        // own chain on failure, so a resize that stops here leaves everything
        // as it was.
        try self.bloom.recreate(.{ .width = hdr.width, .height = hdr.height }, hdr.view);

        self.hdr.deinit();
        self.depth.deinit();
        self.hdr = hdr;
        self.depth = depth;

        post.write(self.context, &self.post_sets, .{
            .view = self.hdr.view,
            .sampler = self.post_sampler,
            .bloom_view = self.bloom.compositeView(),
            .bloom_sampler = self.bloom.compositeSampler(),
        });
    }

    // How many frames have re-recorded the map since init. Diagnostic, and the
    // number a caller checks its own bake policy against: a still scene under a
    // still sun holds this at one.
    pub fn shadowBakes(self: *const Renderer) u64 {
        return self.shadow_bakes;
    }

    pub fn shadowMapSize(self: *const Renderer) u32 {
        return self.shadows.mapSize();
    }

    // How many recordings have drawn a background since init. Every recording
    // that asks for one adds to it, so a caller drawing a background every frame
    // reads its own frame count back: the number says the draw was recorded, not
    // that a picture came out.
    pub fn backgroundDraws(self: *const Renderer) u64 {
        return self.background_draws;
    }

    // How many recordings have run the bloom chain since init. A caller asking
    // for bloom every frame reads its own frame count back; the number says the
    // chain was recorded, not that a glow came out.
    pub fn bloomChains(self: *const Renderer) u64 {
        return self.bloom_chains;
    }

    // How deep the chain built for the current target is. Read by whoever
    // resolves the look, because the composite's weight is normalized against
    // it and a resize can change it.
    pub fn bloomLevels(self: *const Renderer) u32 {
        return self.bloom.levelCount();
    }

    pub fn targetExtent(self: *const Renderer) vk.Extent2D {
        return .{ .width = self.hdr.width, .height = self.hdr.height };
    }

    // The formats the main pass declares, for a caller building a pipeline to
    // record its own draws into it.
    //
    // Read from the attachments rather than named by the caller. Both are the
    // first candidate the device's features carried, so which one it is is not
    // knowable before the device exists, and a pipeline created for a format
    // the pass does not declare cannot be used in it.
    //
    // A resize cannot change them: `resize` recreates the attachments through
    // the same candidate lists on the same device. So a pipeline built from
    // these outlives one, where anything derived from `targetExtent` does not.
    pub fn mainPassFormats(self: *const Renderer) pipeline.Formats {
        return .{ .colour = self.hdr.format, .depth = self.depth.format };
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
    // The camera arrives in the framebuffer's coordinates, which is what its
    // type says and what `uniforms.vulkanClipCamera` is the only producer of.
    // That conversion is the caller's call rather than a rewrite performed here:
    // a value silently altered on the way through is one the caller cannot
    // reason about, and the flip covers two fields, so applying it to one of
    // them is a whole class of defect this signature removes.
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

        try self.frame.update(frame_index, contents);
        self.frame_instance_counts[frame_index] = contents.models.len;
    }

    fn scenePipelineFor(
        self: *const Renderer,
        streams: res.VertexStreams,
        mode: pipeline.Mode,
    ) vk.Pipeline {
        return self.scene_pipelines[scenePipelineIndex(sceneVariantFor(streams), mode)];
    }

    // Every caster in the list, recorded into the open bake.
    //
    // The same batch list the main pass draws, walked in the same order, and
    // that is deliberate rather than convenient: the two must agree about which
    // geometry exists this frame, and a batch drawn from a prepass output has to
    // cast the shape that prepass produced. `batchVertexSource` is what carries
    // the second point, so a morphed mesh casts its blended shape and not its
    // bind pose.
    //
    // The whole list, with no camera culling applied to it. What the sun can see
    // does not depend on where the eye is, and a caster culled out of the camera
    // can still be the one shadowing what the camera does see.
    //
    // Every index below is used without a check and every material record is
    // taken as present. Both hold because the list came out of a `RecordPlan`,
    // which is the only thing `record` accepts and which cannot be made without
    // validating it.
    fn recordCasters(
        self: *const Renderer,
        command_buffer: vk.CommandBuffer,
        frame_index: usize,
        batches: []const RecordBatch,
    ) void {
        if (batches.len == 0) return;
        const device = self.context.device;

        // The bake's layout, not the scene's. They name the same first three
        // set layouts but differ in push constants, so they are not compatible
        // and each pass binds for itself.
        self.frame.bind(self.context, command_buffer, self.shadows.layout, frame_index);

        var last_pipeline: ?vk.Pipeline = null;
        var last_material: ?u32 = null;
        var last_mesh: ?*const mesh_module.Mesh = null;
        var last_source: ?mesh_module.VertexSource = null;

        for (batches) |batch| {
            const material = self.material_records[batch.material_index].?;
            // One axis, not the main pass's two. The bake tests the cutoff
            // against base colour at the mesh's own set-0 UV and reads no
            // material transform, so a masked caster whose base colour slot is
            // transformed casts a silhouette the main pass does not draw.
            const skinned = sceneVariantFor(batch.mesh.streams).skinned;
            // Null is a mode that casts nothing, which is BLEND.
            const variant = shadow.casterVariant(material.alpha, skinned) orelse continue;

            const selected_pipeline = self.shadows.pipelineFor(variant);
            if (last_pipeline == null or last_pipeline.? != selected_pipeline) {
                device.cmdBindPipeline(command_buffer, .graphics, selected_pipeline);
                last_pipeline = selected_pipeline;
            }

            // Only a masked caster reads either of these, and both survive a
            // pipeline change: every pipeline here was built with one layout, so
            // nothing between two masked batches disturbs what the first bound.
            if (variant.masked and (last_material == null or last_material.? != batch.material_index)) {
                device.cmdBindDescriptorSets(
                    command_buffer,
                    .graphics,
                    self.shadows.layout,
                    material_set_index,
                    &.{self.materials.set(@intCast(batch.material_index))},
                    &.{},
                );
                device.cmdPushConstants(
                    command_buffer,
                    self.shadows.layout,
                    shadow.push_constant_range.stage_flags,
                    shadow.push_constant_range.offset,
                    shadow.push_constant_range.size,
                    @ptrCast(&material.mask),
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

    // Draw the background, and count it. The count is what a caller can check
    // from outside: a recording that skipped it and a recording that drew it
    // over nothing look the same from there otherwise.
    fn recordBackground(self: *Renderer, command_buffer: vk.CommandBuffer) void {
        sky.record(self.context, command_buffer, self.sky_pipeline);
        self.background_draws += 1;
    }

    // The renderer's side of `planRecording`: its own state, and the request
    // composition made against it. This is the whole of what recording a frame
    // can fail at.
    pub fn plan(self: *const Renderer, request: RecordRequest) RecordError!RecordPlan {
        return planRecording(.{
            .material_records = self.material_records,
            .frame_instance_counts = self.frame_instance_counts,
            .material_buffer_ready = self.material_buffer_ready,
            .environment_ready = self.environment_ready,
            .bloom_levels = self.bloom.levelCount(),
        }, request);
    }

    // Records the frame the plan describes. Infallible: what could fail happened
    // when the plan was made, and a plan is the only way into this function.
    // The frame in stages, in the order they have to be recorded. Composition
    // calls them; this file no longer states the sequence, because which passes
    // a frame is made of is what composition owns. What each stage still owns
    // is its own precondition, stated on it, so the sequence can be checked
    // against something.
    //
    // Every stage takes the plan `plan` produced and none of them can fail: what
    // could fail happened there, and a plan is the only way into any of them.

    // The sun shadow map. Recorded before `beginMain`, because the map it
    // writes is sampled by every fragment that pass shades, and it opens a
    // rendering of its own that must not be nested in another.
    //
    // Opened when a bake was asked for, and once more when none has happened
    // yet: a lookup must never read an image nothing has written, and the
    // caller cannot be required to remember that its first frame is special.
    //
    // Opening it always records the casters. A pass opened and left empty
    // clears the map to the far plane, which reads as fully lit, and that is
    // a scene without shadows rather than a scene whose shadows are pending.
    // It is the right content for a list with nothing in it to cast, which
    // is what `recordCasters` leaves when it is handed one.
    pub fn recordShadowBake(
        self: *Renderer,
        command_buffer: vk.CommandBuffer,
        shadow_bake: ShadowBake,
        frame_plan: RecordPlan,
    ) void {
        if (shadow_bake != .rebake and self.shadow_valid) return;

        self.shadows.begin(command_buffer);
        // Every batch, not the visible ones: what a camera cannot see still
        // casts into what it can, and the fit this map was built for covers
        // the whole scene rather than the view.
        self.recordCasters(command_buffer, frame_plan.frame_index, frame_plan.casters);
        self.shadows.end(command_buffer);
        self.shadow_valid = true;
        self.shadow_bakes += 1;
    }

    // Opens the main pass: the barriers that order this frame's writes after the
    // previous frame's reads, the dynamic viewport and scissor, and the clear.
    //
    // Nothing is bound here. What a draw inside the pass reads is the draw's
    // own business, and a caller recording its own is not made to inherit
    // bindings it did not ask for.
    pub fn beginMain(self: *Renderer, command_buffer: vk.CommandBuffer) void {
        pass.begin(self.context, command_buffer, self.mainPassTarget(), .{
            .clear_colour = self.clear_colour,
        });
    }

    // The scene: the ordered batch list with the background drawn at its
    // boundary. Recorded between `beginMain` and `endMain`.
    //
    // A caller recording draws of its own does so after this and not before.
    // Binding a descriptor set through a different pipeline layout disturbs what
    // is already bound, so the sets this stage binds have to be the last word
    // inside the pass for the draws that read them.
    pub fn recordScene(
        self: *Renderer,
        command_buffer: vk.CommandBuffer,
        frame_plan: RecordPlan,
    ) void {
        const frame_index = frame_plan.frame_index;
        const batches = frame_plan.batches;
        const background_slot = frame_plan.background_slot;

        const device = self.context.device;

        // The camera and the environment, which the background reads as much as
        // a batch does. One bind serves both because the background's pipeline
        // was built against this same layout.
        if (batches.len > 0 or background_slot != null) {
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
        }

        if (batches.len > 0) {
            // Bound beside the two above and for the same reason: which map a
            // fragment compares against is a property of the frame, not of the
            // batch. The background is not among its readers.
            device.cmdBindDescriptorSets(
                command_buffer,
                .graphics,
                self.scene_layout,
                shadow_set_index,
                &.{self.shadows.descriptorSet()},
                &.{},
            );

            var last_pipeline: ?vk.Pipeline = null;
            var last_material: ?u32 = null;
            var last_cull_mode: ?u32 = null;
            var last_front_face: ?vk.FrontFace = null;
            var last_mesh: ?*const mesh_module.Mesh = null;
            var last_source: ?mesh_module.VertexSource = null;

            for (batches, 0..) |batch, index| {
                if (background_slot == index) {
                    self.recordBackground(command_buffer);
                    // The background's pipeline is what is bound after this, so
                    // the cache no longer describes the command buffer.
                    //
                    // Vulkan specification, VUID-vkCmdDrawIndexed-None-07840:
                    // binding a pipeline that does not declare a dynamic state
                    // invalidates it, and the background declares only viewport
                    // and scissor. So the cull mode is gone as well, and a batch
                    // that happens to want the mode the previous one set would
                    // otherwise skip `vkCmdSetCullMode` and draw with no mode at
                    // all. The winding goes with it, for the same reason and by
                    // the same rule. The descriptor sets and vertex bindings
                    // survive, because the background is created against this
                    // pass's own layout and binds neither.
                    last_pipeline = null;
                    last_cull_mode = null;
                    last_front_face = null;
                }

                // Every material the list names is configured: the plan this
                // list came out of validated it as a whole.
                const mode = self.material_records[batch.material_index].?.mode();
                const selected_pipeline = self.scenePipelineFor(batch.mesh.streams, mode);
                if (last_pipeline == null or last_pipeline.? != selected_pipeline) {
                    device.cmdBindPipeline(command_buffer, .graphics, selected_pipeline);
                    last_pipeline = selected_pipeline;
                }

                const cull_mode = batch.cull_mode.toInt();
                if (last_cull_mode == null or last_cull_mode.? != cull_mode) {
                    device.cmdSetCullMode(command_buffer, batch.cull_mode);
                    last_cull_mode = cull_mode;
                }

                if (last_front_face == null or last_front_face.? != batch.front_face) {
                    device.cmdSetFrontFace(command_buffer, batch.front_face);
                    last_front_face = batch.front_face;
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

        // The slot past the last batch: a list with no blended batch in it has
        // no boundary to reach, and an empty list is that same case with
        // nothing before the background at all.
        if (background_slot == batches.len) self.recordBackground(command_buffer);
    }

    // Closes the main pass and leaves the target in the layout the chain and the
    // post pass sample it in.
    pub fn endMain(self: *Renderer, command_buffer: vk.CommandBuffer) void {
        pass.end(self.context, command_buffer, self.mainPassTarget());
    }

    // The bloom chain, recorded between `endMain` and `recordPost`: it reads
    // what the main pass wrote and the post pass reads what it leaves.
    //
    // No barrier belongs on either side of this. `endMain` has already put the
    // target in the layout the chain's first step samples it in, and the
    // chain's last barrier does the same for its own finest level.
    //
    // A frame whose look asks for no chain skips this, and `recordPost` then
    // takes the pipeline that never names the chain's binding.
    pub fn recordBloom(
        self: *Renderer,
        command_buffer: vk.CommandBuffer,
        frame_plan: RecordPlan,
    ) void {
        const look = frame_plan.look orelse return;
        self.bloom.record(command_buffer, look, frame_plan.post_constants.exposure);
        self.bloom_chains += 1;
    }

    // The pass that presents: the HDR target tone mapped into the image the
    // swapchain handed over. Last, and the only stage that names that image.
    pub fn recordPost(
        self: *Renderer,
        command_buffer: vk.CommandBuffer,
        target: post.Target,
        frame_plan: RecordPlan,
    ) void {
        const look = frame_plan.look;
        const post_constants = frame_plan.post_constants;
        const device = self.context.device;

        post.begin(self.context, command_buffer, target);
        device.cmdBindPipeline(
            command_buffer,
            .graphics,
            if (look == null) self.post_pipeline else self.post_bloom_pipeline,
        );
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

// One pipeline's compiled form, written to the log a line at a time.
//
// Failures are reported and swallowed. This runs only in a build that asked for
// it and produces no value the renderer goes on to use, so refusing to create a
// renderer because a diagnostic could not be gathered would be the instrument
// breaking the thing it measures.
fn reportShaderStatistics(
    context: *const Context,
    allocator: Allocator,
    handle: vk.Pipeline,
    index: usize,
) void {
    const found = pipeline.executables(context, allocator, handle) catch |err| {
        log.warn("scene pipeline {d}: executables unavailable: {t}", .{ index, err });
        return;
    };
    defer allocator.free(found);

    for (found, 0..) |executable, slot| {
        const stats = pipeline.statistics(context, allocator, handle, @intCast(slot)) catch |err| {
            log.warn("scene pipeline {d}: statistics unavailable: {t}", .{ index, err });
            return;
        };
        defer allocator.free(stats);

        log.info("scene pipeline {d}, executable {d}: {s}, subgroup {d}", .{
            index,
            slot,
            pipeline.describedName(&executable.name),
            executable.subgroup_size,
        });
        for (stats) |statistic| {
            const name = pipeline.describedName(&statistic.name);
            switch (statistic.format) {
                .bool32_khr => log.info("  {s}: {}", .{ name, statistic.value.b_32 == .true }),
                .int64_khr => log.info("  {s}: {d}", .{ name, statistic.value.i_64 }),
                .uint64_khr => log.info("  {s}: {d}", .{ name, statistic.value.u_64 }),
                .float64_khr => log.info("  {s}: {d}", .{ name, statistic.value.f_64 }),
                // The enum is open: a driver may report a format this build was
                // compiled before. Naming it beats printing a wrong number.
                _ => log.info("  {s}: unrecognised format {d}", .{ name, @intFromEnum(statistic.format) }),
            }
        }
    }
}

// The post pipelines differ in the fragment entry point and in nothing else, so
// the rest of the configuration is stated once.
fn postConfig(
    layout: vk.PipelineLayout,
    module: vk.ShaderModule,
    shader: post.Shader,
    format: vk.Format,
    fragment_entry: [*:0]const u8,
) pipeline.Config {
    return .{
        .mode = .solid,
        .streams = null,
        .culling = .{ .fixed = .{} },
        .formats = .{ .colour = format },
        .layout = layout,
        .stages = .{
            .vertex = .{ .module = module, .entry_point = shader.vertex_entry },
            .fragment = .{ .module = module, .entry_point = fragment_entry },
        },
    };
}

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
