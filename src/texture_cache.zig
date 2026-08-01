const std = @import("std");
const vk = @import("vulkan");

const Context = @import("context.zig").Context;
const image_module = @import("image.zig");
const ktx2 = @import("ktx2.zig");
const memory = @import("memory/allocator.zig");
const ref_cache = @import("ref_cache.zig");
const sampler_module = @import("sampler.zig");
const staging = @import("staging/arena.zig");

const Allocator = std.mem.Allocator;
const Image = image_module.Image;
const SamplerCache = sampler_module.SamplerCache;
const SamplerConfig = @import("lenore-resources").SamplerConfig;
const StagingArena = staging.StagingArena;

// Vulkan specification, vkCmdCopyBufferToImage: bufferOffset is a multiple of 4
// and of the texel block size. Every format this accepts has 16-byte blocks, and
// KTX2 aligns its levels to the same value, so reserving the whole level block
// at that alignment keeps each level's offset inside it aligned too.
const level_alignment: vk.DeviceSize = 16;

// The neutral image each material slot falls back to when it declares no
// texture. They exist so a shader has no branch: sampling white and multiplying
// by a factor is the factor.
pub const Fallback = enum {
    // Multiplied into base colour and emissive, where white is the identity.
    white,
    // glTF 2.0 specification, 5.22.5: the metallic-roughness texture carries
    // metalness in blue and roughness in green, and red and alpha MUST be
    // ignored for it. Fully rough and not metallic is the neutral surface.
    // Occlusion is a separate texture sampled from red, per 5.19.6, which is why
    // the occlusion slot falls back to white rather than to this.
    metallic_roughness,
    // A tangent-space normal pointing straight out, so no perturbation.
    normal,
    // Keeps a disabled sampled-image binding valid without contributing light.
    black,

    fn texel(self: Fallback) [4]u8 {
        return switch (self) {
            .white => .{ 255, 255, 255, 255 },
            .metallic_roughness => .{ 0, 255, 0, 255 },
            .normal => .{ 128, 128, 255, 255 },
            .black => .{ 0, 0, 0, 255 },
        };
    }

    // Colour is sRGB and data is linear. Reading a normal or a roughness through
    // an sRGB transfer function is a wrong value rather than a wrong shade.
    fn format(self: Fallback) vk.Format {
        return switch (self) {
            .white, .black => .r8g8b8a8_srgb,
            .metallic_roughness, .normal => .r8g8b8a8_unorm,
        };
    }
};

const fallback_count = @typeInfo(Fallback).@"enum".fields.len;

// What a descriptor needs to sample one texture: a view and a sampler, resolved
// separately because an image is deduplicated by content while a sampler is
// chosen per material, so one shared image can be sampled under several wrap
// modes.
//
// Values, never a pointer into the cache. Images live by value in the reference
// cache's hash map, and std/hash_map.zig reallocates its entries when it grows,
// so a stored pointer would dangle as soon as a second texture is loaded. A view
// handle and a sampler handle survive that; the extent is copied for the same
// reason.
pub const Bound = struct {
    view: vk.ImageView,
    sampler: vk.Sampler,
    width: u32,
    height: u32,
    mip_levels: u32,

    fn of(image: *const Image, sampler: vk.Sampler) Bound {
        return .{
            .view = image.view,
            .sampler = sampler,
            .width = image.width,
            .height = image.height,
            .mip_levels = image.mip_levels,
        };
    }
};

pub const InitError = image_module.InitError || staging.ReserveError ||
    image_module.CopyError || Allocator.Error;

pub const AcquireError = error{
    FormatMismatch,
    NotKtx2,
} || ktx2.ParseError || InitError || ref_cache.InsertError;

// Content-addressed store of texture images, with the fallbacks beside it.
//
// The cache owns the images and destroys them when the last reference goes. The
// fallbacks are not in it: they are never released and never deduplicated, and
// keeping them out of the reference count is what makes a slot without a texture
// cost nothing to manage.
pub const TextureCache = struct {
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    host_allocator: Allocator,
    samplers: SamplerCache,
    images: ref_cache.RefCache(Image),
    fallbacks: [fallback_count]Image,

    // Records the fallback uploads into the caller's command buffer rather than
    // submitting them. Nothing here is usable until that buffer has completed,
    // which is the same contract every other resource in this module has.
    pub fn init(
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        host_allocator: Allocator,
        arena: *StagingArena,
        command_buffer: vk.CommandBuffer,
    ) InitError!TextureCache {
        var cache: TextureCache = .{
            .context = context,
            .memory_allocator = memory_allocator,
            .host_allocator = host_allocator,
            .samplers = .init(context),
            .images = .empty,
            .fallbacks = undefined,
        };

        var created: usize = 0;
        errdefer for (cache.fallbacks[0..created]) |*owned| owned.deinit();

        for (std.enums.values(Fallback)) |kind| {
            cache.fallbacks[@intFromEnum(kind)] = try uploadSingleTexel(
                context,
                memory_allocator,
                arena,
                command_buffer,
                kind,
            );
            created += 1;
        }
        return cache;
    }

    // Vulkan specification, vkDestroyImage: submitted work sampling these must
    // have completed. A reference still outstanding is reported by the cache
    // underneath as a leak.
    pub fn deinit(self: *TextureCache) ref_cache.DeinitStatus {
        const status = self.images.deinit(self.host_allocator);
        for (&self.fallbacks) |*owned| owned.deinit();
        self.samplers.deinit(self.host_allocator);
        self.* = undefined;
        return status;
    }

    // A fallback is bound like any other texture. Its image never moves, but it
    // is returned by value all the same so a caller cannot tell the two apart.
    pub fn fallback(
        self: *TextureCache,
        kind: Fallback,
        config: SamplerConfig,
    ) sampler_module.GetError!Bound {
        return .of(&self.fallbacks[@intFromEnum(kind)], try self.sampler(config));
    }

    pub fn sampler(self: *TextureCache, config: SamplerConfig) sampler_module.GetError!vk.Sampler {
        return self.samplers.get(self.host_allocator, config);
    }

    // Takes a reference to the image behind key, uploading it from the given
    // KTX2 bytes on the first request and reusing it afterwards. The bytes need
    // only outlive this call; the copy is recorded against the arena, whose
    // contents must outlive the command buffer instead.
    //
    // The reference this takes must not be dropped before that command buffer is
    // submitted. Releasing the last one destroys an image the recorded copy
    // still names, which invalidates the buffer: it cannot even be ended, let
    // alone submitted. Whoever batches uploads holds the references until the
    // submission completes.
    //
    // The expected format is the one the material slot requires, checked against
    // what the file declares. Accepting a mismatch would interchange normal and
    // colour data silently.
    pub fn acquireKtx2(
        self: *TextureCache,
        key: []const u8,
        bytes: []const u8,
        expected_format: vk.Format,
        sampler_config: SamplerConfig,
        arena: *StagingArena,
        command_buffer: vk.CommandBuffer,
    ) AcquireError!Bound {
        const resolved = try self.sampler(sampler_config);
        if (self.images.acquire(key)) |existing| return .of(existing, resolved);

        if (!ktx2.isKtx2(bytes)) return error.NotKtx2;
        const file = try ktx2.parse(bytes);
        if (file.format != expected_format) return error.FormatMismatch;

        const uploaded = try uploadKtx2(
            self.context,
            self.memory_allocator,
            arena,
            command_buffer,
            bytes,
            file,
        );
        // The cache owns the image from here, including if publishing fails.
        const stored = try self.images.insert(self.host_allocator, key, uploaded);
        return .of(stored, resolved);
    }

    pub fn release(self: *TextureCache, key: []const u8) void {
        self.images.release(self.host_allocator, key);
    }

    pub fn pin(self: *TextureCache, key: []const u8) void {
        self.images.pin(key);
    }

    pub fn count(self: *const TextureCache) u32 {
        return self.images.count();
    }

    pub fn references(self: *const TextureCache, key: []const u8) u32 {
        return self.images.references(key);
    }
};

fn uploadSingleTexel(
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    arena: *StagingArena,
    command_buffer: vk.CommandBuffer,
    kind: Fallback,
) InitError!Image {
    const texel = kind.texel();
    const reservation = try arena.reserve(texel.len, @alignOf(u32));
    @memcpy(reservation.bytes, &texel);

    var image = try Image.init(context, memory_allocator, .{
        .width = 1,
        .height = 1,
        .format = kind.format(),
        .usage = .{ .transfer_dst_bit = true, .sampled_bit = true },
    });
    errdefer image.deinit();

    try recordUpload(&image, arena, command_buffer, &.{.{
        .buffer_offset = reservation.offset,
        .mip_level = 0,
        .width = 1,
        .height = 1,
    }});
    return image;
}

fn uploadKtx2(
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    arena: *StagingArena,
    command_buffer: vk.CommandBuffer,
    bytes: []const u8,
    file: ktx2.File,
) InitError!Image {
    const levels = file.levels();
    // The parser proved the level data is packed without gaps and that the file
    // ends with it, so the whole chain is one span. Copying it as one block
    // keeps each level's offset relative to the block, and the level alignment
    // the parser enforced carries over into the arena.
    const first_byte = levels[levels.len - 1].byte_offset;
    const span = bytes.len - first_byte;

    const reservation = try arena.reserve(span, level_alignment);
    @memcpy(reservation.bytes, bytes[@intCast(first_byte)..]);

    var image = try Image.init(context, memory_allocator, .{
        .width = file.width,
        .height = file.height,
        .format = file.format,
        .mip_levels = @intCast(file.level_count),
        .usage = .{ .transfer_dst_bit = true, .sampled_bit = true },
    });
    errdefer image.deinit();

    var regions: [ktx2.max_levels]image_module.MipCopy = undefined;
    for (levels, 0..) |level, index| {
        regions[index] = .{
            .buffer_offset = reservation.offset + (level.byte_offset - first_byte),
            .mip_level = @intCast(index),
            .width = level.width,
            .height = level.height,
        };
    }

    try recordUpload(&image, arena, command_buffer, regions[0..levels.len]);
    return image;
}

// The three commands every texture upload records, in the order the layouts
// require: receive the copy, copy, then hand the result to shaders.
fn recordUpload(
    image: *Image,
    arena: *StagingArena,
    command_buffer: vk.CommandBuffer,
    regions: []const image_module.MipCopy,
) image_module.CopyError!void {
    image.recordLayoutTransition(command_buffer, .to_transfer_destination);
    try image.recordCopyFrom(arena.source(), command_buffer, regions);
    // Fragment sampling is what a material does with these. A pass that samples
    // a texture from another stage transitions it itself.
    image.recordLayoutTransition(
        command_buffer,
        .toShaderRead(.{ .fragment_shader_bit = true }),
    );
}
