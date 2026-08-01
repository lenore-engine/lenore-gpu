const std = @import("std");
const vk = @import("vulkan");
const Context = @import("context.zig").Context;

const Allocator = std.mem.Allocator;

pub const Filter = enum(u8) { nearest, linear };
pub const MipmapMode = enum(u8) { nearest, linear };
pub const AddressMode = enum(u8) { repeat, mirrored_repeat, clamp_to_edge };

// A Vulkan-free sampler identity, shared by asset import and rendering. It is a
// plain value of enums and a flag, so it hashes and compares field by field:
// that value identity is what makes it usable as a cache key.
//
// Sampler state is deliberately not part of an image's identity. glTF 2.0
// specification, 3.8.2: a texture pairs a source image with a sampler, and
// nothing stops two textures naming the same source with different samplers. An
// image deduplicated by content must therefore be able to carry a different wrap
// mode per material.
pub const SamplerConfig = struct {
    mag_filter: Filter = .linear,
    min_filter: Filter = .linear,
    mipmap_mode: MipmapMode = .linear,
    address_mode_u: AddressMode = .repeat,
    address_mode_v: AddressMode = .repeat,
    address_mode_w: AddressMode = .repeat,
    // Anisotropy is a per-sampler choice rather than a global default nobody can
    // decline. Its cost on this target is unmeasured, so the default is the one
    // that changes nothing about how textures look.
    anisotropic: bool = true,
};

pub const GetError = Allocator.Error || vk.DeviceWrapper.CreateSamplerError;

// Samplers are immutable, cheap, and drawn from a small set: a handful of
// filter and wrap combinations per scene. Each is created once on first request
// and lives until the cache is torn down, rather than one per texture.
pub const SamplerCache = struct {
    context: *const Context,
    map: std.AutoHashMapUnmanaged(SamplerConfig, vk.Sampler) = .empty,

    pub fn init(context: *const Context) SamplerCache {
        return .{ .context = context };
    }

    // Vulkan specification, vkDestroySampler: submitted work using these
    // samplers must have completed.
    pub fn deinit(self: *SamplerCache, allocator: Allocator) void {
        var samplers = self.map.valueIterator();
        while (samplers.next()) |sampler|
            self.context.device.destroySampler(sampler.*, null);
        self.map.deinit(allocator);
        self.* = undefined;
    }

    pub fn get(
        self: *SamplerCache,
        allocator: Allocator,
        config: SamplerConfig,
    ) GetError!vk.Sampler {
        const entry = try self.map.getOrPut(allocator, config);
        if (entry.found_existing) return entry.value_ptr.*;

        errdefer _ = self.map.remove(config);
        entry.value_ptr.* = try self.create(config);
        return entry.value_ptr.*;
    }

    pub fn count(self: *const SamplerCache) u32 {
        return self.map.count();
    }

    fn create(self: *const SamplerCache, config: SamplerConfig) vk.DeviceWrapper.CreateSamplerError!vk.Sampler {
        return self.context.device.createSampler(&.{
            .mag_filter = vulkanFilter(config.mag_filter),
            .min_filter = vulkanFilter(config.min_filter),
            .mipmap_mode = vulkanMipmapMode(config.mipmap_mode),
            .address_mode_u = vulkanAddressMode(config.address_mode_u),
            .address_mode_v = vulkanAddressMode(config.address_mode_v),
            .address_mode_w = vulkanAddressMode(config.address_mode_w),
            .mip_lod_bias = 0.0,
            // Context requires the samplerAnisotropy feature and records the
            // limit this value must not exceed.
            .anisotropy_enable = if (config.anisotropic) .true else .false,
            .max_anisotropy = if (config.anisotropic)
                self.context.max_sampler_anisotropy
            else
                1.0,
            .compare_enable = .false,
            .compare_op = .always,
            .min_lod = 0.0,
            // Vulkan specification, Texel Input Operations: the level of detail
            // used is clamped both by maxLod and by the level count of the view
            // being sampled. Leaving maxLod unclamped therefore makes a sampler
            // independent of any one image's mip depth, which is what lets a
            // single cached sampler serve images with different chains.
            .max_lod = vk.LOD_CLAMP_NONE,
            // No exposed address mode is clamp_to_border, so this is never
            // sampled. It is stated because the field has no default.
            .border_color = .int_opaque_black,
            .unnormalized_coordinates = .false,
        }, null);
    }
};

// Explicit mappings rather than a reflected name bridge. The tag names match
// Vulkan's, and a switch states which pairing is intended instead of resting on
// that coincidence.
fn vulkanFilter(filter: Filter) vk.Filter {
    return switch (filter) {
        .nearest => .nearest,
        .linear => .linear,
    };
}

fn vulkanMipmapMode(mode: MipmapMode) vk.SamplerMipmapMode {
    return switch (mode) {
        .nearest => .nearest,
        .linear => .linear,
    };
}

fn vulkanAddressMode(mode: AddressMode) vk.SamplerAddressMode {
    return switch (mode) {
        .repeat => .repeat,
        .mirrored_repeat => .mirrored_repeat,
        .clamp_to_edge => .clamp_to_edge,
    };
}
