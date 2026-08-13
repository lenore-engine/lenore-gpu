const std = @import("std");
const vk = @import("vulkan");

const Context = @import("../device/context.zig").Context;
const descriptors = @import("../binding/descriptors.zig");
const pass = @import("../pass/scene.zig");
const sampler_module = @import("sampler.zig");
const texture_cache = @import("texture_cache.zig");

const Bound = texture_cache.Bound;
const SamplerConfig = @import("lenore-resources").SamplerConfig;
const TextureCache = texture_cache.TextureCache;

// The image-based half of the scene's lighting: what a fragment reads when it
// asks what the world around it looks like, rather than where the lights are.
//
// Three resources, all prefiltered before the engine sees them. This module owns
// none of them: the texture cache owns the images and this holds the views and
// samplers a descriptor write needs, in the shape the set expects. That split is
// why an environment can be replaced without the renderer knowing how it was
// produced.
//
// The maps are what the Khronos prefilter produces from an environment:
//
//   - lambertian, a cosine-convolved cubemap giving the irradiance arriving at a
//     surface with a given normal. One level; it is smooth by construction.
//   - GGX, a cubemap whose mip levels are roughness steps of the specular
//     integral, sampled along the reflection vector at a level the roughness
//     picks.
//   - a two-channel lookup table for the environment BRDF, tabulated against
//     N.V and roughness. It depends on the BRDF and not on the environment, so
//     it is the same table for every one of them.
//
// The split-sum approximation these three serve is Karis, "Real Shading in
// Unreal Engine 4", SIGGRAPH 2013; the energy compensation the shader applies on
// top is Fdez-Aguera (2019). `shading.zig` holds the host mirror of both.

// The environment's part of the scene set. Slot 0 is the packed material array,
// which `binding/materials.zig` owns; these continue after it, and the renderer
// is where the two lists are joined into one layout. Splitting them this way
// keeps each list next to the code that writes it.
pub const bindings = [_]descriptors.Binding{
    .{ .slot = 1, .name = "lambertian", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
    .{ .slot = 2, .name = "ggx", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
    .{ .slot = 3, .name = "brdf_lut", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
};

const lambertian_slot = bindings[0].slot;
const ggx_slot = bindings[1].slot;
const lut_slot = bindings[2].slot;

// How all three are sampled. Linear between texels and linear between levels:
// the mip chain here is a roughness axis, so the filtering between levels is
// interpolation between two roughnesses and not an anti-aliasing measure.
//
// The address modes are stated but do not decide anything for the two cubemaps.
// Vulkan filters a cube view seamlessly across faces whatever they say, so the
// edge behaviour a 2D texture would get from them is not reachable here. They
// are what the lookup table needs: it is a function tabulated over its domain,
// and a sample at the edge must not wrap to the far side of it.
//
// Anisotropy is off. It buys nothing on a lookup table and nothing on a cubemap
// sampled at a level chosen by roughness rather than by screen-space derivative,
// and it is not free on a tiled sampler.
pub const sampler_config: SamplerConfig = .{
    .mag_filter = .linear,
    .min_filter = .linear,
    .mipmap_mode = .linear,
    .address_mode_u = .clamp_to_edge,
    .address_mode_v = .clamp_to_edge,
    .address_mode_w = .clamp_to_edge,
    .anisotropic = false,
};

// The views and samplers the scene set is written from. A value, not a handle
// into the cache: the same reason `Bound` is one.
pub const Environment = struct {
    lambertian: Bound,
    ggx: Bound,
    lut: Bound,

    // No environment. Every image-based term is linear in the two cubemap
    // samples, so black cubemaps make all of them exactly zero and there is
    // nothing for a flag to switch off. The lookup table is bound black beside
    // them rather than left undefined, because a descriptor a shader can reach
    // has to name a real image whether or not the value is used.
    pub fn neutral(cache: *TextureCache) sampler_module.GetError!Environment {
        return .{
            .lambertian = try cache.fallback(.black_cube, sampler_config),
            .ggx = try cache.fallback(.black_cube, sampler_config),
            .lut = try cache.fallback(.black, sampler_config),
        };
    }
};

// Points the scene set at an environment. Cold: this runs when an environment is
// loaded or replaced, never per frame.
//
// Vulkan specification, vkUpdateDescriptorSets: the set must not be in use by
// any submitted work that has not completed. Replacing an environment while a
// frame is in flight is therefore the caller's problem, not this function's.
pub fn write(context: *const Context, set: vk.DescriptorSet, source: Environment) void {
    const images = [_]vk.DescriptorImageInfo{
        imageInfo(source.lambertian),
        imageInfo(source.ggx),
        imageInfo(source.lut),
    };

    var writes: [bindings.len]vk.WriteDescriptorSet = undefined;
    for (bindings, &images, &writes) |binding, *info, *entry| {
        entry.* = .{
            .dst_set = set,
            .dst_binding = binding.slot,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = binding.kind,
            .p_image_info = @ptrCast(info),
            .p_buffer_info = &no_buffers,
            .p_texel_buffer_view = &no_texel_buffers,
        };
    }
    context.device.updateDescriptorSets(&writes, null);
}

fn imageInfo(bound: Bound) vk.DescriptorImageInfo {
    return .{
        .sampler = bound.sampler,
        .image_view = bound.view,
        // Every texture reaching a descriptor has already been transitioned by
        // the upload that filled it.
        .image_layout = pass.sampled_layout,
    };
}

// Vulkan specification, VkWriteDescriptorSet: the members not selected by
// descriptorType are ignored, but the pointers are not optional in the
// structure, so they are given something valid to point at.
const no_buffers = [_]vk.DescriptorBufferInfo{};
const no_texel_buffers = [_]vk.BufferView{};
