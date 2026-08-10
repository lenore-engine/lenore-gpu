const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");
const res = @import("lenore-resources");

const testing = std.testing;

// A limit no device reports, so a value that reaches the create info can only
// have come from the argument.
const device_anisotropy: f32 = 12.5;

test "sampler: a minification filter without mipmapping clamps to the base level" {
    // glTF 2.0 specification, 3.8.4.2: NEAREST and LINEAR sample the original
    // image. Vulkan has no mipmap mode for that, so the request arrives as a
    // level-of-detail ceiling of zero and the mode itself is free to be the
    // cheap one. TextureTransformTest is the asset that asks for this.
    const info = gpu.samplerCreateInfo(.{ .mipmap_mode = .none }, device_anisotropy);
    try testing.expectEqual(@as(f32, 0.0), info.max_lod);
    try testing.expectEqual(@as(f32, 0.0), info.min_lod);
    try testing.expectEqual(vk.SamplerMipmapMode.nearest, info.mipmap_mode);
}

test "sampler: a mipmapped config leaves the ceiling to the image" {
    // The unclamped ceiling is what lets one cached sampler serve images with
    // different chain depths, so both mipmapped modes must reach it.
    for ([_]res.MipmapMode{ .nearest, .linear }) |mode| {
        const info = gpu.samplerCreateInfo(.{ .mipmap_mode = mode }, device_anisotropy);
        try testing.expectEqual(vk.LOD_CLAMP_NONE, info.max_lod);
    }

    try testing.expectEqual(
        vk.SamplerMipmapMode.nearest,
        gpu.samplerCreateInfo(.{ .mipmap_mode = .nearest }, device_anisotropy).mipmap_mode,
    );
    try testing.expectEqual(
        vk.SamplerMipmapMode.linear,
        gpu.samplerCreateInfo(.{ .mipmap_mode = .linear }, device_anisotropy).mipmap_mode,
    );
}

test "sampler: each filter and address mode reaches its own Vulkan value" {
    // The two filter axes are checked with opposite values, so a mapping that
    // read the wrong field would show up rather than agree with itself.
    const info = gpu.samplerCreateInfo(.{
        .mag_filter = .nearest,
        .min_filter = .linear,
        .address_mode_u = .clamp_to_edge,
        .address_mode_v = .mirrored_repeat,
        .address_mode_w = .repeat,
    }, device_anisotropy);

    try testing.expectEqual(vk.Filter.nearest, info.mag_filter);
    try testing.expectEqual(vk.Filter.linear, info.min_filter);
    try testing.expectEqual(vk.SamplerAddressMode.clamp_to_edge, info.address_mode_u);
    try testing.expectEqual(vk.SamplerAddressMode.mirrored_repeat, info.address_mode_v);
    try testing.expectEqual(vk.SamplerAddressMode.repeat, info.address_mode_w);
}

test "sampler: anisotropy is the device limit or nothing" {
    const on = gpu.samplerCreateInfo(.{ .anisotropic = true }, device_anisotropy);
    try testing.expectEqual(vk.TRUE, @intFromEnum(on.anisotropy_enable));
    try testing.expectEqual(device_anisotropy, on.max_anisotropy);

    // One rather than the limit: with anisotropy off the value asks for nothing,
    // and writing the device's number there would read as a request.
    const off = gpu.samplerCreateInfo(.{ .anisotropic = false }, device_anisotropy);
    try testing.expectEqual(vk.FALSE, @intFromEnum(off.anisotropy_enable));
    try testing.expectEqual(@as(f32, 1.0), off.max_anisotropy);
}

test "sampler: the environment's own config is unaffected by the clamp" {
    // environment.sampler_config samples a prefiltered cubemap by roughness, so
    // its chain is the whole point and its ceiling must stay unclamped.
    const info = gpu.samplerCreateInfo(gpu.environmentSampler, device_anisotropy);
    try testing.expectEqual(vk.LOD_CLAMP_NONE, info.max_lod);
    try testing.expectEqual(vk.SamplerMipmapMode.linear, info.mipmap_mode);
}
