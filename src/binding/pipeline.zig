const std = @import("std");
const build_options = @import("build_options");
const vk = @import("vulkan");
const res = @import("lenore-resources");
const Context = @import("../device/context.zig").Context;
const vertex = @import("../object/vertex.zig");

const Allocator = std.mem.Allocator;
const VertexStreams = res.VertexStreams;
const AlphaMode = res.MaterialInfo.Rendering.AlphaMode;

pub const CreateError = vk.DeviceWrapper.CreateShaderModuleError ||
    vk.DeviceWrapper.CreatePipelineLayoutError ||
    vk.DeviceWrapper.CreateGraphicsPipelinesError ||
    vk.DeviceWrapper.CreateComputePipelinesError;

// How a primitive reaches the colour target. They differ in more than blending:
// a blended primitive does not write depth, because it is drawn after the opaque
// set in an order that is only approximately back to front, and depth writes
// would let one transparent surface hide another.
pub const Mode = enum {
    solid,
    blended,

    // The background, drawn at the far plane between the two sets above. It
    // composites by the depth test alone: nothing blends, and nothing is
    // written back, because the only value it could write is the one the clear
    // already put there.
    //
    // Not a mode any material maps to. `modeFor` cannot produce it, and the
    // table of scene pipelines names its own two rather than counting the
    // members here.
    background,

    // The bloom chain's upsample: the filtered coarser level summed onto the
    // level under it. The shader has already scaled its result by that level's
    // weight, so the blend unit's whole part is the sum.
    //
    // The one mode with no depth in it at all. The chain is colour only, and
    // where the others answer what a primitive does about depth, this one
    // answers that there is none to do anything about.
    additive,
};

// glTF's third alpha mode has no pipeline of its own. A masked primitive
// discards in the shader against the material's cutoff and is an opaque draw
// otherwise, so it shares the solid pipeline.
pub fn modeFor(alpha: AlphaMode) Mode {
    return switch (alpha) {
        .@"opaque", .mask => .solid,
        .blend => .blended,
    };
}

// Dynamic rendering takes the attachment formats at creation instead of a
// render pass object.
pub const Formats = struct {
    // Null for a pass that writes no colour. The shadow bake is one: the depth
    // the rasterizer writes is its whole output, and declaring an attachment
    // nothing writes would demand a fragment shader to write it.
    colour: ?vk.Format = null,
    // Null for a pass that declares no depth attachment. The post pass is one:
    // it covers the screen with a triangle and has nothing to occlude.
    depth: ?vk.Format = null,
};

// A created shader module together with the entry point to take from it. One
// module carries several: a Slang file marked up with `[shader(...)]`
// attributes emits every entry point it declares into one binary, so the two
// stages below routinely name the same module and differ only here.
pub const Stage = struct {
    module: vk.ShaderModule,
    entry_point: [*:0]const u8,
};

pub const Stages = struct {
    vertex: Stage,
    // Null for a pipeline that writes no colour and needs no per-fragment test.
    // The opaque half of the shadow bake is one, and leaving the stage out is
    // not merely an economy: a fragment shader is what would stand between the
    // rasterizer and the depth write.
    fragment: ?Stage = null,
};

// The two factors the rasterizer offsets a fragment's depth by, held together
// because neither is meaningful without depth bias being enabled at all. Null
// in `Config` is the disabled state, so there is no enable flag beside values
// that would then be ignored.
//
// Vulkan specification, Rasterization, "Depth Bias Computation"
// (primsrast-depthbias-computation): the slope factor scales the maximum depth
// slope of the polygon, the constant factor scales the depth attachment's
// parameter r, and the two scaled terms are summed. Both are passed through
// unchanged.
//
// The two are not interchangeable, and the reason lands on whoever picks the
// numbers. The slope term is in units of the primitive's own depth gradient.
// The constant term is in units of r, which for a floating-point depth
// attachment the same section defines per primitive as 2^(e - n), where e is the
// maximum exponent over the z values the primitive spans; there is no single
// minimum resolvable difference for such an attachment.
pub const DepthBias = struct {
    constant: f32 = 0,
    slope: f32,
};

// Cull mode is fixed for a pass whose geometry does not vary by material and
// dynamic for the scene pass, where a double-sided material disables it per
// draw. Keeping that distinction in pipeline state prevents a later pipeline
// from inheriting a dynamic mode set for an earlier one.
pub const Culling = union(enum) {
    fixed: vk.CullModeFlags,
    dynamic,
};

pub const Config = struct {
    mode: Mode,
    // Null for a pipeline whose vertex stage reads no buffer. The fullscreen
    // triangle is one: its positions come from the vertex index, and declaring
    // a binding nothing binds is a draw-time error rather than a create-time
    // one.
    streams: ?VertexStreams,
    culling: Culling,
    formats: Formats,
    layout: vk.PipelineLayout,
    stages: Stages,
    // Null leaves the rasterizer's depth untouched, which is what every pass
    // writing into the camera's depth buffer wants: the value it writes is the
    // one it is later tested against.
    depth_bias: ?DepthBias = null,
};

// Whether the driver is asked to keep what it compiled a pipeline into, so
// `statistics` can be asked for it afterwards.
//
// One answer for the whole build rather than a field on every configuration:
// which pipelines an instrument reads is not a property of any one of them, and
// a per-pipeline flag would let a build capture some and then be asked about the
// rest. Vulkan specification, VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR: the
// statistics exist only for a pipeline created with the flag, and setting it may
// cost pipeline creation time, so a build asks for it with `-Dshader-stats` when
// someone is going to read the answer.
pub const capture_statistics = build_options.capture_shader_statistics;

// The whole set of streams: base, skin, colour and second UV, contributing four
// attributes, two, one and one.
pub const max_bindings = 4;
pub const max_attributes = 8;

pub const VertexInput = struct {
    bindings: [max_bindings]vk.VertexInputBindingDescription,
    binding_count: u32,
    attributes: [max_attributes]vk.VertexInputAttributeDescription,
    attribute_count: u32,

    pub fn boundStreams(self: *const VertexInput) []const vk.VertexInputBindingDescription {
        return self.bindings[0..self.binding_count];
    }

    pub fn declaredAttributes(self: *const VertexInput) []const vk.VertexInputAttributeDescription {
        return self.attributes[0..self.attribute_count];
    }
};

// The bindings and attributes for exactly the streams a mesh carries. The
// binding numbers and locations are the vertex types' own, so this and
// `Mesh.bind` agree by reading the same declarations rather than by two lists
// kept in step.
pub fn vertexInput(streams: VertexStreams) VertexInput {
    var input: VertexInput = .{
        .bindings = undefined,
        .binding_count = 0,
        .attributes = undefined,
        .attribute_count = 0,
    };

    // The base stream is not optional: a mesh with no position is not a mesh.
    addStream(&input, vertex.GpuVertex);
    if (streams.skinned) addStream(&input, vertex.GpuSkinVertex);
    if (streams.colour) addStream(&input, vertex.GpuColourVertex);
    if (streams.uv1) addStream(&input, vertex.GpuUv1Vertex);

    return input;
}

fn addStream(input: *VertexInput, comptime Stream: type) void {
    input.bindings[input.binding_count] = Stream.binding_description;
    input.binding_count += 1;
    for (Stream.attribute_descriptions) |attribute| {
        input.attributes[input.attribute_count] = attribute;
        input.attribute_count += 1;
    }
}

// Every mode drawn into the camera's target tests depth against what the opaque
// set already wrote. Only the solid one adds to it: a blended surface that wrote
// depth would hide the blended surfaces behind it, which is the one thing
// sorting cannot repair.
//
// The background is the one that has to accept equality. It is drawn at the far
// plane, and the pixels it belongs in are exactly those still holding the clear,
// which is that same value; a strict comparison would reject all of them and
// draw nothing at all.
//
// The additive mode states no depth behaviour because it has none. `create`
// hands this structure to the driver only when the rendering info declares a
// depth attachment, and the bloom chain declares none.
pub fn depthStencilState(mode: Mode) vk.PipelineDepthStencilStateCreateInfo {
    return .{
        .depth_test_enable = if (mode == .additive) .false else .true,
        .depth_write_enable = if (mode == .solid) .true else .false,
        .depth_compare_op = switch (mode) {
            .solid, .blended => .less,
            .background => .less_or_equal,
            .additive => .always,
        },
        .depth_bounds_test_enable = .false,
        .stencil_test_enable = .false,
        .front = std.mem.zeroes(vk.StencilOpState),
        .back = std.mem.zeroes(vk.StencilOpState),
        .min_depth_bounds = 0,
        .max_depth_bounds = 1,
    };
}

// Source alpha over destination, with the destination's alpha left alone: the
// HDR target is opaque and its alpha is composited against nothing.
//
// The additive mode is the exception in the colour factors, and it reads no
// alpha at all: what the bloom chain's upsample wants is the plain sum of the
// tent-filtered coarser level and the level already in the attachment. Its
// weight is not here either. It is a uniform the shader applies, so the value
// stays where a test can evaluate it rather than becoming blend state that
// nothing offline can read back.
//
// The modes that do not blend still carry factors. A disabled blend ignores
// them, and giving each its own set would be several states to read where the
// enable bit is the only difference that reaches the device.
pub fn blendAttachment(mode: Mode) vk.PipelineColorBlendAttachmentState {
    return .{
        .blend_enable = switch (mode) {
            .blended, .additive => .true,
            .solid, .background => .false,
        },
        .src_color_blend_factor = switch (mode) {
            .additive => .one,
            .solid, .blended, .background => .src_alpha,
        },
        .dst_color_blend_factor = switch (mode) {
            .additive => .one,
            .solid, .blended, .background => .one_minus_src_alpha,
        },
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .zero,
        .alpha_blend_op = .add,
        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
    };
}

pub fn createModule(context: *const Context, spirv: []const u32) CreateError!vk.ShaderModule {
    // Vulkan specification, VkShaderModuleCreateInfo: the code is a multiple of
    // four bytes and four-byte aligned. Taking words rather than bytes makes
    // both true by construction.
    return context.device.createShaderModule(&.{
        .code_size = spirv.len * @sizeOf(u32),
        .p_code = spirv.ptr,
    }, null);
}

pub const LayoutConfig = struct {
    descriptor_sets: []const vk.DescriptorSetLayout = &.{},
    push_constants: []const vk.PushConstantRange = &.{},
};

pub fn createLayout(context: *const Context, config: LayoutConfig) CreateError!vk.PipelineLayout {
    return context.device.createPipelineLayout(&.{
        .set_layout_count = @intCast(config.descriptor_sets.len),
        .p_set_layouts = config.descriptor_sets.ptr,
        .push_constant_range_count = @intCast(config.push_constants.len),
        .p_push_constant_ranges = config.push_constants.ptr,
    }, null);
}

// glTF 2.0, section 3.7.4 "Instantiation": a positive-determinant node has
// counter-clockwise winding. Vulkan determines facing from signed area in
// framebuffer coordinates (Basic Polygon Rasterization); `vulkanClip` negates
// clip y before the positive-height viewport maps it to framebuffer rows, so
// that winding is `counter_clockwise` here. Draws with a negative determinant
// must reverse the culled side or use the opposite front-face state.
pub fn rasterizationState(
    culling: Culling,
    bias: ?DepthBias,
) vk.PipelineRasterizationStateCreateInfo {
    return .{
        .depth_clamp_enable = .false,
        .rasterizer_discard_enable = .false,
        .polygon_mode = .fill,
        .cull_mode = switch (culling) {
            .fixed => |mode| mode,
            .dynamic => .{},
        },
        .front_face = .counter_clockwise,
        .depth_bias_enable = if (bias == null) .false else .true,
        .depth_bias_constant_factor = if (bias) |values| values.constant else 0,
        // Unclamped. A clamp bounds the offset a steep primitive receives, and
        // what it is worth cannot be settled without a scene steep enough to
        // need one.
        .depth_bias_clamp = 0,
        .depth_bias_slope_factor = if (bias) |values| values.slope else 0,
        .line_width = 1,
    };
}

const fixed_dynamic_states = [_]vk.DynamicState{ .viewport, .scissor };
// The winding travels with the culled side. A draw whose transform mirrors it
// needs the opposite front face whether or not it culls anything, because the
// fragment stage learns which side it is shading from the same state, so a
// pipeline that can vary one varies both. Both commands are required by
// `VK_GRAPHICS_VERSION_1_3` in vk.xml, so neither asks for anything this device
// was not already built with.
const facing_dynamic_states = fixed_dynamic_states ++
    [_]vk.DynamicState{ .cull_mode, .front_face };

// A fixed pipeline ignores the cull mode and winding left by an earlier dynamic
// draw. A dynamic one must receive `vkCmdSetCullMode` and `vkCmdSetFrontFace`
// before each draw whose state can differ.
pub fn dynamicStates(culling: Culling) []const vk.DynamicState {
    return switch (culling) {
        .fixed => &fixed_dynamic_states,
        .dynamic => &facing_dynamic_states,
    };
}

pub fn create(context: *const Context, config: Config) CreateError!vk.Pipeline {
    var stages: [2]vk.PipelineShaderStageCreateInfo = undefined;
    stages[0] = .{
        .stage = .{ .vertex_bit = true },
        .module = config.stages.vertex.module,
        .p_name = config.stages.vertex.entry_point,
    };
    var stage_count: u32 = 1;
    if (config.stages.fragment) |fragment| {
        stages[1] = .{
            .stage = .{ .fragment_bit = true },
            .module = fragment.module,
            .p_name = fragment.entry_point,
        };
        stage_count = 2;
    }

    const input = if (config.streams) |streams| vertexInput(streams) else VertexInput{
        .bindings = undefined,
        .binding_count = 0,
        .attributes = undefined,
        .attribute_count = 0,
    };
    const vertex_input_state = vk.PipelineVertexInputStateCreateInfo{
        .vertex_binding_description_count = input.binding_count,
        .p_vertex_binding_descriptions = &input.bindings,
        .vertex_attribute_description_count = input.attribute_count,
        .p_vertex_attribute_descriptions = &input.attributes,
    };

    const input_assembly = vk.PipelineInputAssemblyStateCreateInfo{
        .topology = .triangle_list,
        .primitive_restart_enable = .false,
    };

    // Counts only. The values are dynamic state and come from the pass, which
    // is what lets one pipeline survive a resize.
    const viewport_state = vk.PipelineViewportStateCreateInfo{
        .viewport_count = 1,
        .scissor_count = 1,
    };

    const rasterization = rasterizationState(config.culling, config.depth_bias);

    const multisample = vk.PipelineMultisampleStateCreateInfo{
        .rasterization_samples = .{ .@"1_bit" = true },
        .sample_shading_enable = .false,
        .min_sample_shading = 1,
        .alpha_to_coverage_enable = .false,
        .alpha_to_one_enable = .false,
    };

    const depth_stencil = depthStencilState(config.mode);
    const blend_attachment = blendAttachment(config.mode);

    // A pipeline with no colour attachment has nothing to blend, and the state
    // is still handed over with a count of zero rather than left out: a valid
    // structure describing no attachments is accepted wherever the state is
    // read at all, and it is one branch fewer than reasoning about when the
    // pointer may be null.
    const blend_state = vk.PipelineColorBlendStateCreateInfo{
        .logic_op_enable = .false,
        .logic_op = .copy,
        .attachment_count = if (config.formats.colour == null) 0 else 1,
        .p_attachments = @ptrCast(&blend_attachment),
        .blend_constants = .{ 0, 0, 0, 0 },
    };

    const dynamic_states = dynamicStates(config.culling);
    const dynamic_state = vk.PipelineDynamicStateCreateInfo{
        .dynamic_state_count = @intCast(dynamic_states.len),
        .p_dynamic_states = dynamic_states.ptr,
    };

    // Named rather than passed inline: the structure holds a pointer to it and
    // it has to outlive the call below.
    const colour_format = config.formats.colour orelse .undefined;
    const rendering = vk.PipelineRenderingCreateInfo{
        .view_mask = 0,
        .color_attachment_count = if (config.formats.colour == null) 0 else 1,
        .p_color_attachment_formats = @ptrCast(&colour_format),
        .depth_attachment_format = config.formats.depth orelse .undefined,
        .stencil_attachment_format = .undefined,
    };

    var pipeline: vk.Pipeline = undefined;
    _ = try context.device.createGraphicsPipelines(.null_handle, &.{.{
        .p_next = @ptrCast(&rendering),
        .flags = .{ .capture_statistics_bit_khr = capture_statistics },
        .stage_count = stage_count,
        .p_stages = &stages,
        .p_vertex_input_state = &vertex_input_state,
        .p_input_assembly_state = &input_assembly,
        .p_viewport_state = &viewport_state,
        .p_rasterization_state = &rasterization,
        .p_multisample_state = &multisample,
        // Vulkan specification, VkGraphicsPipelineCreateInfo: the depth stencil
        // state is read only when the rendering info declares a depth or
        // stencil attachment, and this pipeline may declare neither.
        .p_depth_stencil_state = if (config.formats.depth == null) null else &depth_stencil,
        .p_color_blend_state = &blend_state,
        .p_dynamic_state = &dynamic_state,
        .layout = config.layout,
        .render_pass = .null_handle,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    }}, null, (&pipeline)[0..1]);

    return pipeline;
}

pub const ComputeConfig = struct {
    layout: vk.PipelineLayout,
    stage: Stage,
};

// A compute pipeline is one stage and a layout. None of the state above applies:
// there is no vertex input, no rasterizer and no attachment, so the config that
// selects them has no compute counterpart rather than a set of ignored fields.
//
// The workgroup size is not here either. Vulkan takes it from the module's
// LocalSize execution mode, so it is declared in the shader and a dispatch has to
// divide by the same number; see the morph prepass, which holds the two together.
pub fn createCompute(context: *const Context, config: ComputeConfig) CreateError!vk.Pipeline {
    var pipeline: vk.Pipeline = undefined;
    _ = try context.device.createComputePipelines(.null_handle, &.{.{
        .flags = .{ .capture_statistics_bit_khr = capture_statistics },
        .stage = .{
            .stage = .{ .compute_bit = true },
            .module = config.stage.module,
            .p_name = config.stage.entry_point,
        },
        .layout = config.layout,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    }}, null, (&pipeline)[0..1]);

    return pipeline;
}

// What the driver compiled a pipeline into, as VK_KHR_pipeline_executable_properties
// reports it. One pipeline may hold several executables: a graphics pipeline on
// this class of device usually has one per stage, and a driver is free to merge
// or split them, so the count is asked for rather than assumed.
//
// Only a pipeline created with `Config.capture_statistics` has any of this, and
// only a device with the extension can be asked at all. Both are the caller's to
// establish; `Context.shader_statistics_enabled` answers the second.
pub const StatisticsError = vk.DeviceWrapper.GetPipelineExecutablePropertiesKHRError ||
    vk.DeviceWrapper.GetPipelineExecutableStatisticsKHRError ||
    Allocator.Error;

// A name or description as the specification declares it: a fixed array padded
// with zeroes rather than a slice. Trimmed at the first zero, and whole when
// there is none.
pub fn describedName(field: []const u8) []const u8 {
    const end = std.mem.indexOfScalar(u8, field, 0) orelse field.len;
    return field[0..end];
}

// The executables of one pipeline. The caller owns the slice.
pub fn executables(
    context: *const Context,
    allocator: Allocator,
    handle: vk.Pipeline,
) StatisticsError![]vk.PipelineExecutablePropertiesKHR {
    const info = vk.PipelineInfoKHR{ .pipeline = handle };
    var count: u32 = 0;
    _ = try context.device.getPipelineExecutablePropertiesKHR(&info, &count, null);
    const result = try allocator.alloc(vk.PipelineExecutablePropertiesKHR, count);
    errdefer allocator.free(result);
    // Every entry is written by the driver, and the struct carries a type tag
    // the caller never sets. Cleared rather than left undefined so that a driver
    // writing fewer entries than it counted leaves zeroes and not stack noise.
    @memset(result, .{
        .stages = .{},
        .name = @splat(0),
        .description = @splat(0),
        .subgroup_size = 0,
    });
    _ = try context.device.getPipelineExecutablePropertiesKHR(&info, &count, result.ptr);
    return result[0..count];
}

// One executable's statistics: on this driver the register counts and the size
// of the compiled code. What is reported is the driver's choice, so nothing here
// names an individual statistic.
pub fn statistics(
    context: *const Context,
    allocator: Allocator,
    handle: vk.Pipeline,
    executable_index: u32,
) StatisticsError![]vk.PipelineExecutableStatisticKHR {
    const info = vk.PipelineExecutableInfoKHR{
        .pipeline = handle,
        .executable_index = executable_index,
    };
    var count: u32 = 0;
    _ = try context.device.getPipelineExecutableStatisticsKHR(&info, &count, null);
    const result = try allocator.alloc(vk.PipelineExecutableStatisticKHR, count);
    errdefer allocator.free(result);
    @memset(result, .{
        .name = @splat(0),
        .description = @splat(0),
        .format = .uint64_khr,
        .value = .{ .u_64 = 0 },
    });
    _ = try context.device.getPipelineExecutableStatisticsKHR(&info, &count, result.ptr);
    return result[0..count];
}
