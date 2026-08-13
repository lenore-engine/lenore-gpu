const std = @import("std");
const vk = @import("vulkan");

const Context = @import("../device/context.zig").Context;
const image_module = @import("image.zig");
const ktx2 = @import("ktx2.zig");
const memory = @import("../memory/allocator.zig");
const ref_cache = @import("../store/ref_cache.zig");
const retirement = @import("../store/retirement.zig");
const sampler_module = @import("sampler.zig");
const transfer_module = @import("../staging/transfer.zig");

const log = std.log.scoped(.vulkan);

const Allocator = std.mem.Allocator;
const Image = image_module.Image;
const SamplerCache = sampler_module.SamplerCache;
const SamplerConfig = @import("lenore-resources").SamplerConfig;
const Transfer = transfer_module.Transfer;

// Decoded pixels are four bytes a texel, tightly packed.
const rgba8_texel_bytes: vk.DeviceSize = 4;

// Vulkan specification, vkCmdCopyBufferToImage: bufferOffset is a multiple of 4
// and of the texel block size. An RGBA8 texel block is one texel, so its own
// size satisfies both. A KTX2 file reports its value instead, which differs
// between a 16-byte compressed block and an 8-byte texel.
const rgba8_alignment: vk.DeviceSize = rgba8_texel_bytes;

// Decoded pixels are one texel per block, so a row of them is a row of blocks.
const uncompressed_block_height: u32 = 1;

// Decoded pixels, independent of the container or decoder that produced them.
// Rows are tightly packed from the top of the source image, four bytes per
// texel. The bytes need only outlive acquisition; they are copied into staging.
pub const Rgba8 = struct {
    width: u32,
    height: u32,
    bytes: []const u8,
};

pub const Rgba8Error = error{
    InvalidExtent,
    PixelLengthMismatch,
    PixelLengthOverflow,
    UnsupportedPixelFormat,
};

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
    // The environment when there is none. Unlike the emissive slot, where an
    // absent texture means white and the presence mask has to say so, an absent
    // environment really is zero radiance from every direction: the image-based
    // terms are linear in the two cubemap samples, so black makes all of them
    // vanish exactly rather than approximately. That is why this one needs no
    // flag beside it.
    black_cube,

    fn texel(self: Fallback) [4]u8 {
        return switch (self) {
            .white => .{ 255, 255, 255, 255 },
            .metallic_roughness => .{ 0, 255, 0, 255 },
            .normal => .{ 128, 128, 255, 255 },
            .black, .black_cube => .{ 0, 0, 0, 255 },
        };
    }

    // Colour is sRGB and data is linear. Reading a normal or a roughness through
    // an sRGB transfer function is a wrong value rather than a wrong shade. An
    // environment is radiance and so is linear too, which black does not reveal
    // but the format still has to state.
    pub fn format(self: Fallback) vk.Format {
        return switch (self) {
            .white, .black => .r8g8b8a8_srgb,
            .metallic_roughness, .normal, .black_cube => .r8g8b8a8_unorm,
        };
    }

    pub fn shape(self: Fallback) image_module.Shape {
        return switch (self) {
            .white, .metallic_roughness, .normal, .black => .texture_2d,
            .black_cube => .cube,
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
// The cache's half of a binding: which image, without which sampler. The two
// are cached separately, so a second use that wants different filtering pairs
// this with another sampler instead of acquiring the image again.
//
// A value rather than a pointer into the image storage, for the same reason
// Bound is: the storage keeps images by value and rehashes when it grows.
pub const Resident = struct {
    view: vk.ImageView,
    width: u32,
    height: u32,
    mip_levels: u32,

    fn of(image: *const Image) Resident {
        return .{
            .view = image.view,
            .width = image.width,
            .height = image.height,
            .mip_levels = image.mip_levels,
        };
    }

    // Nothing is acquired here. The caller holds the reference the image was
    // acquired under, and it must outlive every binding made from it.
    pub fn bind(self: Resident, sampler: vk.Sampler) Bound {
        return .{
            .view = self.view,
            .sampler = sampler,
            .width = self.width,
            .height = self.height,
            .mip_levels = self.mip_levels,
        };
    }
};

pub const Bound = struct {
    view: vk.ImageView,
    sampler: vk.Sampler,
    width: u32,
    height: u32,
    mip_levels: u32,

    fn of(image: *const Image, sampler: vk.Sampler) Bound {
        return Resident.of(image).bind(sampler);
    }

    // The image half of this binding, for a use that pairs it with a sampler of
    // its own.
    pub fn resident(self: Bound) Resident {
        return .{
            .view = self.view,
            .width = self.width,
            .height = self.height,
            .mip_levels = self.mip_levels,
        };
    }
};

pub const InitError = image_module.InitError || transfer_module.ReserveError ||
    image_module.CopyError || Allocator.Error;

pub const AcquireError = error{
    FormatMismatch,
    KeyConflict,
    KindMismatch,
    NotKtx2,
    // The faces handed in do not describe six squares of the declared extent at
    // the declared format's texel size.
    CubeBytesMismatch,
} || Rgba8Error || ktx2.ParseError || InitError || ref_cache.InsertError;

// The container's notion of what the file is, and the image type that has to be
// created for it. They are separate enums because the parser has no Vulkan in
// it, and this is the one place the two meet. Exhaustive switches catch a
// missing case; only a test catches the two being swapped, which is why this is
// public despite having one caller.
pub fn imageShape(kind: ktx2.Kind) image_module.Shape {
    return switch (kind) {
        .texture_2d => .texture_2d,
        .cube => .cube,
    };
}

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
    // Borrowed, not owned: the frame ring it is keyed to belongs to whoever
    // drives the frame loop, and so does the queue.
    retirement: *retirement.ResourceRetirement,

    // Records the fallback uploads into the caller's transfer rather than
    // submitting them. Nothing here is usable until that transfer has finished,
    // which is the same contract every other resource in this module has.
    pub fn init(
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        host_allocator: Allocator,
        transfer: *Transfer,
        image_retirement: *retirement.ResourceRetirement,
    ) InitError!TextureCache {
        var cache: TextureCache = .{
            .context = context,
            .memory_allocator = memory_allocator,
            .host_allocator = host_allocator,
            .samplers = .init(context),
            .images = .empty,
            .fallbacks = undefined,
            .retirement = image_retirement,
        };

        var created: usize = 0;
        errdefer for (cache.fallbacks[0..created]) |*owned| rollback(owned, transfer);

        for (std.enums.values(Fallback)) |kind| {
            cache.fallbacks[@intFromEnum(kind)] = try uploadSingleTexel(
                context,
                memory_allocator,
                transfer,
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
    // only outlive this call, because they are copied into staging as they are
    // consumed.
    //
    // The reference this takes must not be dropped before the transfer has
    // finished. Releasing the last one destroys an image a recorded copy still
    // names, which invalidates the command buffer: it cannot even be ended, let
    // alone submitted. Whoever batches uploads holds the references until the
    // submission completes.
    //
    // The expected format and kind are what the binding requires, checked against
    // what the file declares. Accepting a format mismatch would interchange
    // normal and colour data silently. Accepting a kind mismatch is worse than
    // silent: a cube image bound where the shader declared a 2D sampler, or the
    // reverse, is a descriptor the device rejects, and the message names the
    // binding rather than the file that was wrong.
    pub fn acquireKtx2(
        self: *TextureCache,
        key: []const u8,
        bytes: []const u8,
        expected_format: vk.Format,
        expected_kind: ktx2.Kind,
        sampler_config: SamplerConfig,
        transfer: *Transfer,
    ) AcquireError!Bound {
        if (!ktx2.isKtx2(bytes)) return error.NotKtx2;
        const file = try ktx2.parse(bytes);
        if (file.format != expected_format) return error.FormatMismatch;
        if (file.kind != expected_kind) return error.KindMismatch;

        const resolved = try self.sampler(sampler_config);
        if (try self.acquireExisting(key, .{
            .format = file.format,
            .shape = imageShape(file.kind),
            .width = file.width,
            .height = file.height,
            .mip_levels = @intCast(file.level_count),
        }, resolved)) |existing| return existing;

        const uploaded = try uploadKtx2(
            self.context,
            self.memory_allocator,
            transfer,
            bytes,
            file,
        );
        // The cache owns the image from here, including if publishing fails.
        const stored = try self.images.insert(self.host_allocator, key, uploaded);
        return .of(stored, resolved);
    }

    // Uploads one decoded RGBA8 level. Decoding is deliberately outside the GPU
    // module; this path consumes the same plain pixels whether they came from
    // PNG, JPEG, an editor canvas or generated content.
    //
    // The image has one mip. Vulkan clamps sampling to the view's level count,
    // so this is complete without pretending that mip generation and its colour
    // filtering policy have been decided here.
    // A cube of one level, from faces the caller already holds.
    //
    // Separate from `acquireKtx2` because the source is not a file: a caller that
    // has resampled a level has bytes and an extent and nothing else, and going
    // back through a container would mean writing one to read it again.
    //
    // The faces are six squares, contiguous, in the order a cube level stores
    // them. `texel_bytes` is the format's, and it is a parameter rather than a
    // table lookup so that this call cannot silently disagree with what the
    // caller packed.
    pub fn acquireCube(
        self: *TextureCache,
        key: []const u8,
        faces: []const u8,
        extent: u32,
        texel_bytes: u32,
        format: vk.Format,
        sampler_config: SamplerConfig,
        transfer: *Transfer,
    ) AcquireError!Bound {
        const face_texels = @as(usize, extent) * extent;
        if (extent == 0 or faces.len != ktx2.cube_faces * face_texels * texel_bytes)
            return error.CubeBytesMismatch;

        const resolved = try self.sampler(sampler_config);
        if (try self.acquireExisting(key, .{
            .format = format,
            .shape = .cube,
            .width = extent,
            .height = extent,
            .mip_levels = 1,
        }, resolved)) |existing| return existing;

        const uploaded = try uploadCube(
            self.context,
            self.memory_allocator,
            transfer,
            faces,
            extent,
            texel_bytes,
            format,
        );
        const stored = try self.images.insert(self.host_allocator, key, uploaded);
        return .of(stored, resolved);
    }

    pub fn acquireRgba8(
        self: *TextureCache,
        key: []const u8,
        source: Rgba8,
        format: vk.Format,
        sampler_config: SamplerConfig,
        transfer: *Transfer,
    ) AcquireError!Bound {
        try validateRgba8(source, format);

        const resolved = try self.sampler(sampler_config);
        if (try self.acquireExisting(key, .{
            .format = format,
            .shape = .texture_2d,
            .width = source.width,
            .height = source.height,
            .mip_levels = 1,
        }, resolved)) |existing| return existing;

        const uploaded = try uploadRgba8(
            self.context,
            self.memory_allocator,
            transfer,
            source,
            format,
        );
        const stored = try self.images.insert(self.host_allocator, key, uploaded);
        return .of(stored, resolved);
    }

    // Host-side validation exposed on the type so container decoders can be
    // tested without constructing a device or a cache.
    pub fn validateRgba8(source: Rgba8, format: vk.Format) Rgba8Error!void {
        return validateRgba8Source(source, format);
    }

    const ExpectedImage = struct {
        format: vk.Format,
        shape: image_module.Shape,
        width: u32,
        height: u32,
        mip_levels: u32,
    };

    fn acquireExisting(
        self: *TextureCache,
        key: []const u8,
        expected: ExpectedImage,
        resolved_sampler: vk.Sampler,
    ) AcquireError!?Bound {
        const existing = self.images.acquire(key) orelse return null;
        if (existing.format != expected.format or
            existing.shape != expected.shape or
            existing.width != expected.width or
            existing.height != expected.height or
            existing.mip_levels != expected.mip_levels)
        {
            // `acquire` already took a reference. Undo it before reporting that
            // the caller reused one identity for different image content.
            //
            // The undo cannot be what drops the last reference, since the entry
            // was present before this call, so nothing normally comes back. It
            // is destroyed rather than discarded because discarding it would
            // leak an image on the strength of that argument alone.
            if (self.images.release(self.host_allocator, key)) |returned| {
                var image = returned;
                image.deinit();
            }
            return error.KeyConflict;
        }
        return .of(existing, resolved_sampler);
    }

    // Vulkan specification, vkDestroyImage and vkDestroyImageView: submitted work
    // using either must have completed. The image therefore goes to the frame
    // ring rather than to the device here, and the caller is free to release a
    // texture at any point in a frame without knowing what is in flight.
    //
    // Infallible on purpose: this is reached from teardown paths that cannot
    // report anything. What a queue that will not take the image falls back to
    // is stated once, at `retirement.retireOrDestroy`.
    pub fn release(self: *TextureCache, key: []const u8) void {
        const returned = self.images.release(self.host_allocator, key) orelse return;
        retirement.retireOrDestroy(self.retirement, self.host_allocator, .{ .image = returned });
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

fn validateRgba8Source(source: Rgba8, format: vk.Format) Rgba8Error!void {
    if (source.width == 0 or source.height == 0) return error.InvalidExtent;
    if (format != .r8g8b8a8_srgb and format != .r8g8b8a8_unorm)
        return error.UnsupportedPixelFormat;

    const width = std.math.cast(usize, source.width) orelse return error.PixelLengthOverflow;
    const height = std.math.cast(usize, source.height) orelse return error.PixelLengthOverflow;
    const pixels = std.math.mul(usize, width, height) catch return error.PixelLengthOverflow;
    const expected = std.math.mul(usize, pixels, 4) catch return error.PixelLengthOverflow;
    if (source.bytes.len != expected) return error.PixelLengthMismatch;
}

fn uploadSingleTexel(
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    transfer: *Transfer,
    kind: Fallback,
) InitError!Image {
    const texel = kind.texel();
    const shape = kind.shape();

    var image = try Image.init(context, memory_allocator, .{
        .width = 1,
        .height = 1,
        .format = kind.format(),
        .usage = .{ .transfer_dst_bit = true, .sampled_bit = true },
        .shape = shape,
    });
    errdefer rollback(&image, transfer);

    image.recordLayoutTransition(transfer.commandBuffer(), .to_transfer_destination);
    // A cube's six faces are the same texel, and each is its own copy because
    // every copy names one layer.
    for (0..shape.layerCount()) |layer| {
        try uploadFace(&image, transfer, &texel, .{
            .mip_level = 0,
            .layer = @intCast(layer),
            .height = 1,
            .block_height = uncompressed_block_height,
            .row_bytes = texel.len,
            .alignment = rgba8_alignment,
        });
    }
    recordShaderRead(&image, transfer);
    return image;
}

fn uploadRgba8(
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    transfer: *Transfer,
    source: Rgba8,
    format: vk.Format,
) InitError!Image {
    var image = try Image.init(context, memory_allocator, .{
        .width = source.width,
        .height = source.height,
        .format = format,
        .usage = .{ .transfer_dst_bit = true, .sampled_bit = true },
    });
    errdefer rollback(&image, transfer);

    image.recordLayoutTransition(transfer.commandBuffer(), .to_transfer_destination);
    // validateRgba8Source proved the byte count is exactly width by height by
    // four, so this row divides the face a whole number of times.
    try uploadFace(&image, transfer, source.bytes, .{
        .mip_level = 0,
        .layer = 0,
        .height = source.height,
        .block_height = uncompressed_block_height,
        .row_bytes = @as(vk.DeviceSize, source.width) * rgba8_texel_bytes,
        .alignment = rgba8_alignment,
    });
    recordShaderRead(&image, transfer);
    return image;
}

fn uploadKtx2(
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    transfer: *Transfer,
    bytes: []const u8,
    file: ktx2.File,
) InitError!Image {
    var image = try Image.init(context, memory_allocator, .{
        .width = file.width,
        .height = file.height,
        .format = file.format,
        .mip_levels = @intCast(file.level_count),
        .usage = .{ .transfer_dst_bit = true, .sampled_bit = true },
        .shape = imageShape(file.kind),
    });
    errdefer rollback(&image, transfer);

    image.recordLayoutTransition(transfer.commandBuffer(), .to_transfer_destination);
    for (file.levels(), 0..) |level, index| {
        // The parser proved each level's data lies inside the file and that a
        // level holds face_count faces of face_byte_length each, consecutively.
        for (0..file.face_count) |face| {
            const start = level.byte_offset + face * level.face_byte_length;
            try uploadFace(
                &image,
                transfer,
                bytes[@intCast(start)..][0..@intCast(level.face_byte_length)],
                .{
                    .mip_level = @intCast(index),
                    .layer = @intCast(face),
                    .height = level.height,
                    .block_height = file.block_height,
                    .row_bytes = level.row_bytes,
                    .alignment = file.level_alignment,
                },
            );
        }
    }
    recordShaderRead(&image, transfer);
    return image;
}

// One face of one mip level, as the copy below needs to see it.
// The same shape as `uploadKtx2` with the container's answers supplied by the
// caller: one level, six faces, no block compression, so a row of texel blocks
// is a row of texels and every face is tightly packed.
fn uploadCube(
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    transfer: *Transfer,
    faces: []const u8,
    extent: u32,
    texel_bytes: u32,
    format: vk.Format,
) InitError!Image {
    var image = try Image.init(context, memory_allocator, .{
        .width = extent,
        .height = extent,
        .format = format,
        .mip_levels = 1,
        .usage = .{ .transfer_dst_bit = true, .sampled_bit = true },
        .shape = .cube,
    });
    errdefer rollback(&image, transfer);

    image.recordLayoutTransition(transfer.commandBuffer(), .to_transfer_destination);
    const face_bytes = @as(usize, extent) * extent * texel_bytes;
    for (0..ktx2.cube_faces) |face| {
        try uploadFace(
            &image,
            transfer,
            faces[face * face_bytes ..][0..face_bytes],
            .{
                .mip_level = 0,
                .layer = @intCast(face),
                .height = extent,
                .block_height = 1,
                .row_bytes = @as(u64, extent) * texel_bytes,
                // Nothing was placed by a container, so the only alignment a
                // staged piece has to keep is the one its own texels need.
                .alignment = texel_bytes,
            },
        );
    }
    recordShaderRead(&image, transfer);
    return image;
}

const FaceGeometry = struct {
    mip_level: u32,
    layer: u32,
    // Texel rows of this level, which is what bounds the range a copy states.
    height: u32,
    // Texel rows spanned by one row of texel blocks.
    block_height: u32,
    // Bytes of one row of texel blocks.
    row_bytes: u64,
    alignment: vk.DeviceSize,
};

// Copies one tightly packed face into the image through the staging pool, a run
// of whole rows at a time.
//
// Rows of texel blocks are the unit because they are the largest run of a face
// that is both contiguous in the source and expressible as a copy: a rectangle
// is neither. That is what lets a 48 MiB cube face travel through 256 KiB of
// staging without the pool having to be sized for it.
fn uploadFace(
    image: *const Image,
    transfer: *Transfer,
    face: []const u8,
    geometry: FaceGeometry,
) InitError!void {
    const row_bytes = geometry.row_bytes;
    const block_height: u64 = geometry.block_height;
    const block_rows = @as(u64, face.len) / row_bytes;

    var written: u64 = 0;
    while (written < block_rows) {
        const reservation = try transfer.reserve(.{
            .size = (block_rows - written) * row_bytes,
            .alignment = geometry.alignment,
            .granularity = row_bytes,
        });
        @memcpy(
            reservation.bytes,
            face[@intCast(written * row_bytes)..][0..reservation.bytes.len],
        );

        // Exact: the pool returns either the whole request or a multiple of the
        // granularity, and both are multiples of a row.
        const rows = @as(u64, reservation.bytes.len) / row_bytes;
        // Below the level height because written is below the block row count,
        // so the subtraction that follows cannot go negative.
        const first_texel_row: u32 = @intCast(written * block_height);
        // The last run of block rows can overhang the level, because a level's
        // height need not be a multiple of the block height. Clamping is what
        // keeps the copy inside the image, and it is the only place the stated
        // extent is not a whole number of blocks.
        const texel_rows: u32 = @intCast(@min(
            @as(u64, geometry.height - first_texel_row),
            rows * block_height,
        ));

        try image.recordCopyFrom(reservation.source, transfer.commandBuffer(), .{
            .buffer_offset = reservation.offset,
            .mip_level = geometry.mip_level,
            .layer = geometry.layer,
            .first_row = first_texel_row,
            .row_count = texel_rows,
        });
        written += rows;
    }
}

// Fragment sampling is what a material does with these. A pass that samples a
// texture from another stage transitions it itself.
fn recordShaderRead(image: *Image, transfer: *Transfer) void {
    image.recordLayoutTransition(
        transfer.commandBuffer(),
        .toShaderRead(.{ .fragment_shader_bit = true }),
    );
}

// Destroying an image that a submission may still be reading is forbidden, and
// a submission that failed after reaching the queue establishes neither
// completion nor that it never started. Vulkan specification, vkDestroyImage. A
// leak is recoverable and a use after free is not.
fn rollback(image: *Image, transfer: *const Transfer) void {
    if (transfer.abandoned()) {
        std.log.err(
            "texture upload abandoned after a failed submission: its image is " ++
                "leaked because the copies naming it may still be pending",
            .{},
        );
        return;
    }
    image.deinit();
}
