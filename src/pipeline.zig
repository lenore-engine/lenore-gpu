const std = @import("std");
const vk = @import("vulkan");
const res = @import("lenore-resources");
const Context = @import("context.zig").Context;
const vertex = @import("mesh/vertex.zig");

const VertexStreams = res.VertexStreams;
const AlphaMode = res.MaterialInfo.Rendering.AlphaMode;

pub const CreateError = vk.DeviceWrapper.CreateShaderModuleError ||
    vk.DeviceWrapper.CreatePipelineLayoutError ||
    vk.DeviceWrapper.CreateGraphicsPipelinesError;

// How a primitive reaches the colour target. The two differ in more than
// blending: a blended primitive does not write depth, because it is drawn after
// the opaque set in an order that is only approximately back to front, and
// depth writes would let one transparent surface hide another.
pub const Mode = enum {
    solid,
    blended,
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
    colour: vk.Format,
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
    fragment: Stage,
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
};

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

// Both modes test depth against what the opaque set already wrote. Only the
// solid one adds to it: a blended surface that wrote depth would hide the
// blended surfaces behind it, which is the one thing sorting cannot repair.
pub fn depthStencilState(mode: Mode) vk.PipelineDepthStencilStateCreateInfo {
    return .{
        .depth_test_enable = .true,
        .depth_write_enable = if (mode == .solid) .true else .false,
        .depth_compare_op = .less,
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
// The factors are set in both modes. A disabled blend ignores them, and giving
// the solid mode its own set of values would be two states to read where the
// enable bit is the only difference that reaches the device.
pub fn blendAttachment(mode: Mode) vk.PipelineColorBlendAttachmentState {
    return .{
        .blend_enable = if (mode == .blended) .true else .false,
        .src_color_blend_factor = .src_alpha,
        .dst_color_blend_factor = .one_minus_src_alpha,
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
pub fn rasterizationState(culling: Culling) vk.PipelineRasterizationStateCreateInfo {
    return .{
        .depth_clamp_enable = .false,
        .rasterizer_discard_enable = .false,
        .polygon_mode = .fill,
        .cull_mode = switch (culling) {
            .fixed => |mode| mode,
            .dynamic => .{},
        },
        .front_face = .counter_clockwise,
        .depth_bias_enable = .false,
        .depth_bias_constant_factor = 0,
        .depth_bias_clamp = 0,
        .depth_bias_slope_factor = 0,
        .line_width = 1,
    };
}

const fixed_dynamic_states = [_]vk.DynamicState{ .viewport, .scissor };
const cull_dynamic_states = fixed_dynamic_states ++ [_]vk.DynamicState{.cull_mode};

// A fixed pipeline ignores cull mode left by an earlier dynamic draw. A dynamic
// one must receive `vkCmdSetCullMode` before each draw whose mode can differ.
pub fn dynamicStates(culling: Culling) []const vk.DynamicState {
    return switch (culling) {
        .fixed => &fixed_dynamic_states,
        .dynamic => &cull_dynamic_states,
    };
}

pub fn create(context: *const Context, config: Config) CreateError!vk.Pipeline {
    const stages = [_]vk.PipelineShaderStageCreateInfo{
        .{
            .stage = .{ .vertex_bit = true },
            .module = config.stages.vertex.module,
            .p_name = config.stages.vertex.entry_point,
        },
        .{
            .stage = .{ .fragment_bit = true },
            .module = config.stages.fragment.module,
            .p_name = config.stages.fragment.entry_point,
        },
    };

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

    const rasterization = rasterizationState(config.culling);

    const multisample = vk.PipelineMultisampleStateCreateInfo{
        .rasterization_samples = .{ .@"1_bit" = true },
        .sample_shading_enable = .false,
        .min_sample_shading = 1,
        .alpha_to_coverage_enable = .false,
        .alpha_to_one_enable = .false,
    };

    const depth_stencil = depthStencilState(config.mode);
    const blend_attachment = blendAttachment(config.mode);

    const blend_state = vk.PipelineColorBlendStateCreateInfo{
        .logic_op_enable = .false,
        .logic_op = .copy,
        .attachment_count = 1,
        .p_attachments = @ptrCast(&blend_attachment),
        .blend_constants = .{ 0, 0, 0, 0 },
    };

    const dynamic_states = dynamicStates(config.culling);
    const dynamic_state = vk.PipelineDynamicStateCreateInfo{
        .dynamic_state_count = @intCast(dynamic_states.len),
        .p_dynamic_states = dynamic_states.ptr,
    };

    const colour_format = config.formats.colour;
    const rendering = vk.PipelineRenderingCreateInfo{
        .view_mask = 0,
        .color_attachment_count = 1,
        .p_color_attachment_formats = @ptrCast(&colour_format),
        .depth_attachment_format = config.formats.depth orelse .undefined,
        .stencil_attachment_format = .undefined,
    };

    var pipeline: vk.Pipeline = undefined;
    _ = try context.device.createGraphicsPipelines(.null_handle, &.{.{
        .p_next = @ptrCast(&rendering),
        .stage_count = stages.len,
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
