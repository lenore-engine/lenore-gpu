const std = @import("std");
const vk = @import("vulkan");

const Context = @import("../device/context.zig").Context;
const descriptors = @import("../binding/descriptors.zig");
const sampler_module = @import("sampler.zig");
const texture_cache = @import("texture_cache.zig");

const Bound = texture_cache.Bound;
const SamplerConfig = @import("lenore-resources").SamplerConfig;
const TextureCache = texture_cache.TextureCache;

// Light that was computed somewhere other than this frame, stored per surface.
//
// Two textures for a whole scene, addressed by the second UV set. One holds the
// irradiance arriving at each patch of static surface; the other holds signed
// coefficients that correct it when a shading normal differs from the geometric
// normal. Irradiance is the same quantity
// the lambertian environment carries and it answers the same question, with the
// difference that it knows about this scene: an environment cube says what the
// world outside sends a surface with a given normal, and cannot know that a wall
// stands between them.
//
// The engine neither fills this nor says how it was filled. A loader may hand
// over an image baked offline, and a pass may hand over one it is still writing
// into; both reach the shader down this one binding, and nothing downstream can
// tell which it was.
//
// **The texel at (0, 0) must be black.** A mesh with no second UV set has a
// zero coordinate in it, and a sampler clamped at the edge answers that with
// exactly that texel. So the surfaces that store no light read the corner, and
// the corner has to add nothing. The fallback below is black everywhere and
// satisfies this by construction; a producer of a real atlas has to leave the
// corner clear, which for a packer with any margin at all it already does.

// The lighting cache's part of the scene set. Slots 0 through 3 belong to the
// packed material array and the environment; the renderer is where the lists are
// joined, and a slot claimed twice is a compile error there.
pub const bindings = [_]descriptors.Binding{
    .{ .slot = 4, .name = "lightmap", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
    .{ .slot = 5, .name = "lightmap_direction", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
};

// Bilinear between texels and no mip chain.
//
// The chain is absent rather than unused: a lower level would average texels
// from charts that are neighbours in the atlas and nowhere near each other in
// the world, which is the one artefact a surface lightmap has to be built to
// avoid. Charts carry a margin for the same reason, and a mip level is wide
// enough to cross it.
//
// Clamped on both axes, which is what makes the corner rule above hold: a
// coordinate of zero is half a texel outside the image and resolves to the
// texel at the origin rather than wrapping to the far side.
//
// Anisotropy is off. The sample is a lookup at a surface's own parameterisation,
// not a projection of a pattern across it at a grazing angle.
pub const sampler_config: SamplerConfig = .{
    .mag_filter = .linear,
    .min_filter = .linear,
    .mipmap_mode = .nearest,
    .address_mode_u = .clamp_to_edge,
    .address_mode_v = .clamp_to_edge,
    .address_mode_w = .clamp_to_edge,
    .anisotropic = false,
};

// The view and sampler the scene set is written from. A value, not a handle into
// the cache, for the reason `Bound` is one.
pub const Lightmap = struct {
    irradiance: Bound,
    direction: Bound,

    // No stored light. Black irradiance makes the whole term zero, and the
    // directional fallback decodes to zero coefficients, which makes the
    // directional factor one. The latter is also the fallback for a scalar-only
    // cache, so adding this binding does not change how an existing irradiance
    // atlas shades. The two are different images because neutral is a different
    // texel in each: coefficients are stored biased, so black is not zero there.
    pub fn none(cache: *TextureCache) sampler_module.GetError!Lightmap {
        return .{
            .irradiance = try cache.fallback(.black, sampler_config),
            .direction = try cache.fallback(.directional, sampler_config),
        };
    }

    pub fn scalar(cache: *TextureCache, irradiance: Bound) sampler_module.GetError!Lightmap {
        return .{
            .irradiance = irradiance,
            .direction = try cache.fallback(.directional, sampler_config),
        };
    }
};

// Points the scene set at a lighting cache. Cold: when one is loaded or
// replaced, never per frame. A pass that keeps writing into the same image does
// not come back here, because the descriptor names the image and not its
// contents.
//
// Vulkan specification, vkUpdateDescriptorSets: the set must not be in use by
// any submitted work that has not completed. Handing over a different image
// while a frame is in flight is therefore the caller's problem.
pub fn write(context: *const Context, set: vk.DescriptorSet, source: Lightmap) void {
    const infos = [_]vk.DescriptorImageInfo{
        imageInfo(source.irradiance),
        imageInfo(source.direction),
    };
    var writes: [bindings.len]vk.WriteDescriptorSet = undefined;
    for (&writes, bindings, 0..) |*destination, binding, index| {
        destination.* = .{
            .dst_set = set,
            .dst_binding = binding.slot,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = binding.kind,
            .p_image_info = @ptrCast(&infos[index]),
            .p_buffer_info = &no_buffers,
            .p_texel_buffer_view = &no_texel_buffers,
        };
    }
    context.device.updateDescriptorSets(&writes, null);
}

fn imageInfo(source: Bound) vk.DescriptorImageInfo {
    return .{
        .sampler = source.sampler,
        .image_view = source.view,
        // Every texture reaching a descriptor has already been transitioned by
        // whatever filled it.
        .image_layout = .shader_read_only_optimal,
    };
}

const no_buffers = [_]vk.DescriptorBufferInfo{};
const no_texel_buffers = [_]vk.BufferView{};
