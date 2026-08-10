const std = @import("std");
const vk = @import("vulkan");

const Context = @import("context.zig").Context;
const pipeline = @import("pipeline.zig");

// The background: three vertices at the far plane, drawn wherever the opaque
// set left the depth buffer untouched. What they shade is the caller's, and
// this file has no opinion on it.
//
// It owns nothing. There is no image, no sampler and no descriptor set here,
// because everything a background can want is already bound for the draws on
// either side of it: the camera in the frame set, the environment in the scene
// set. What is left is a pipeline built against the layout the scene draws
// through, so those sets serve this draw as they stand and it costs one bind
// and three vertices.
//
// Sharing the scene's sets rather than declaring its own is what keeps a
// background and the surfaces on one radiometric scale by construction. Two
// copies of an environment, or one copy behind two intensities, is a background
// that drifts out of agreement with the lighting and no test can say by how
// much. The absent environment needs no branch either, for the reason
// `environment.zig` gives: the neutral cube is black, and a black background is
// the picture a scene without an environment should have.

// Whether a recording draws one. Composition's to decide per frame, and not
// retained here for the reason `post.Settings` is not: what has already been
// submitted must not change when a later frame asks for something else.
//
// `clear` is not "no sky". It is the main pass's clear colour left standing,
// which is what a caller wanting a flat field behind an asset asks for, and the
// diagnostic that tells a pass which drew nothing from a post chain which
// carried nothing.
pub const Background = enum {
    clear,
    environment,
};

// The three vertices of the covering triangle, generated from the vertex index.
// It reads no buffer, so the pipeline declares no vertex input.
pub const vertex_count: u32 = 3;

// What a background shader has to supply for this pass to be built from it.
//
// Both entry points are required and neither is optional. A background with no
// fragment stage is not a pass this file can record, so an absent name would
// buy a check on every creation path in exchange for a state that has no
// meaning. Words that arrive from outside the module are what makes that
// distinction worth drawing: an optional here would be unwrapped against data
// this module did not author.
//
// The words are free to read whatever the scene layout binds and nothing else.
// A shader declaring a set of its own cannot be created against the layout
// passed to `config`, which is where the mismatch surfaces.
pub const Shader = struct {
    spirv: []const u32,
    vertex_entry: [*:0]const u8,
    fragment_entry: [*:0]const u8,
};

// The pipeline the background is drawn with.
//
// The layout is the scene's, passed in rather than created here. Every set the
// shader reads is one the main pass binds for the draws on either side of it,
// and a layout of its own would differ in nothing while making the two
// incompatible: binding for one would then disturb what the other had bound.
//
// Culling is fixed at none rather than dynamic. The scene draws around this one
// set a cull mode per batch, and a pipeline that declared the state dynamic
// would inherit whichever batch went last.
pub fn config(
    layout: vk.PipelineLayout,
    module: vk.ShaderModule,
    shader: Shader,
    formats: pipeline.Formats,
) pipeline.Config {
    return .{
        .mode = .background,
        .streams = null,
        .culling = .{ .fixed = .{} },
        .formats = formats,
        .layout = layout,
        .stages = .{
            .vertex = .{ .module = module, .entry_point = shader.vertex_entry },
            .fragment = .{ .module = module, .entry_point = shader.fragment_entry },
        },
    };
}

// Draw it. Recorded inside the main pass, between the opaque batches and the
// blended ones: the opaques are what the depth test rejects it against, and the
// blended set composites over whatever it leaves.
//
// The caller has bound the frame and scene sets. Nothing is bound here, because
// what this draw reads is what the batches around it read.
pub fn record(context: *const Context, command_buffer: vk.CommandBuffer, background: vk.Pipeline) void {
    context.device.cmdBindPipeline(command_buffer, .graphics, background);
    context.device.cmdDraw(command_buffer, vertex_count, 1, 0, 0);
}
