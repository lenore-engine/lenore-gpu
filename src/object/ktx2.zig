const std = @import("std");
const vk = @import("vulkan");

// A narrow KTX2 reader for texture artifacts, safe against hostile input: every
// field is validated before anything is created, and the parser touches no
// allocator and no Vulkan object.
//
// Two shapes are accepted, and they are accepted for different reasons. A 2D
// single-face block-compressed image is what a shipped game loads, and it must
// carry a complete baked mip chain. A six-face uncompressed cube map is a
// prefiltered environment, whose level count is chosen by the prefilter rather
// than by the extent. Layout from the KTX File Format Specification v2: a fixed
// 80-byte header, then one 24-byte level index entry per level, all
// little-endian.

const identifier = [12]u8{ 0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A };
const header_size: u64 = 80;
const level_index_entry_size: u64 = 24;

// KTX File Format Specification v2, cube maps: a cube has six faces in the
// +X, -X, +Y, -Y, +Z, -Z order Vulkan also uses for its array layers.
pub const cube_faces: u32 = 6;

// The same cap the image type enforces on a mip chain, which describes extents
// up to 32768 texels. Holding it here keeps parsing allocation-free.
pub const max_levels: usize = 16;

// What the file is, which decides both the chain rule below and the kind of
// Vulkan image the caller has to create for it.
pub const Kind = enum {
    texture_2d,
    cube,
};

// Khronos Data Format Specification v1.3, table "Color models": the block
// compressed models are 128 plus the BC index, so BC5 is 132 and BC7 is 134.
// Transfer functions in the same document: KHR_DF_TRANSFER_LINEAR is 1 and
// KHR_DF_TRANSFER_SRGB is 2.
const Descriptor = struct {
    colour_model: u8,
    transfer_function: u8,
};

const Supported = struct {
    format: vk.Format,
    // KTX File Format Specification v2, header: the size of one channel of one
    // texel, and 1 for a block-compressed format.
    type_size: u32,
    // One texel for an uncompressed format, so the level arithmetic below is
    // written once and does not branch on whether the format has blocks.
    block_width: u32,
    block_height: u32,
    block_bytes: u64,
    // What the data format descriptor must say for this format, or null where
    // the descriptor cannot be trusted. See `validateDataFormat`.
    descriptor: ?Descriptor,
};

// Adding a format means adding its geometry in the same row, so a format can
// never be accepted without the arithmetic that decides its level sizes.
const supported_formats = [_]Supported{
    .{
        .format = .bc5_unorm_block,
        .type_size = 1,
        .block_width = 4,
        .block_height = 4,
        .block_bytes = 16,
        .descriptor = .{ .colour_model = 132, .transfer_function = 1 },
    },
    .{
        .format = .bc7_unorm_block,
        .type_size = 1,
        .block_width = 4,
        .block_height = 4,
        .block_bytes = 16,
        .descriptor = .{ .colour_model = 134, .transfer_function = 1 },
    },
    .{
        .format = .bc7_srgb_block,
        .type_size = 1,
        .block_width = 4,
        .block_height = 4,
        .block_bytes = 16,
        .descriptor = .{ .colour_model = 134, .transfer_function = 2 },
    },
    // The format every prefiltered environment Khronos publishes is stored in.
    // Its descriptor is null because those files do not carry a conformant one:
    // the section is present and 28 bytes long, but declares dfdTotalSize 0,
    // colour model 0, no samples and bytesPlane entirely zero, written that way
    // by "UX3D SlimKTX2 v1.0". KTX File Format Specification v2 section 3.9.2
    // requires dfdByteLength to equal dfdTotalSize, so those files are invalid
    // as they stand, and requiring a conformant descriptor here would reject the
    // reference environments themselves.
    //
    // Taking the format from vkFormat alone loses nothing. Section 3.10.1
    // requires the descriptor's texel block dimensions, bytesPlane and sample
    // information to match the format's definition whenever vkFormat is not
    // VK_FORMAT_UNDEFINED, so a conformant descriptor restates vkFormat and
    // cannot qualify it. R16G16B16A16_SFLOAT and R16G16B16A16_UNORM are distinct
    // vkFormat values, 97 and 91, so telling those two apart was never the
    // descriptor's work. That is what makes the check in `validateDataFormat` a
    // consistency test against the row it matched, not the source of the format.
    .{
        .format = .r16g16b16a16_sfloat,
        .type_size = 2,
        .block_width = 1,
        .block_height = 1,
        .block_bytes = 8,
        .descriptor = null,
    },
};

pub const ParseError = error{
    InvalidDataFormat,
    InvalidDimensions,
    InvalidIdentifier,
    InvalidLevel,
    InvalidMipChain,
    MissingMipChain,
    OverlappingData,
    TooManyLevels,
    Truncated,
    UnsupportedFormat,
    UnsupportedLayout,
    UnsupportedSupercompression,
};

pub const Level = struct {
    // Absolute offset into the parsed bytes.
    byte_offset: u64,
    // Every face of this level. The specification's levelImages holds
    // layers by faces by depth images and this length spans all of them, so for
    // a cube map it is six times the size of one face.
    byte_length: u64,
    // One face of this level, which is what a copy covers when it fits in one
    // piece.
    face_byte_length: u64,
    // One row of texel blocks of one face. A face too large to copy at once is
    // split in multiples of this, because that is the largest run of its bytes
    // that is contiguous and describes whole rows.
    row_bytes: u64,
    width: u32,
    height: u32,
};

// The parse result is a value. The level cap makes the chain a fixed array, so
// there is nothing to free and nothing to outlive the call.
pub const File = struct {
    format: vk.Format,
    kind: Kind,
    face_count: u32,
    width: u32,
    height: u32,
    // What every level offset in this file is a multiple of, and what an uploader
    // has to place each piece it stages at, or the offsets it hands the copy stop
    // being aligned.
    level_alignment: u64,
    // Texel rows spanned by one row of texel blocks: four for a block compressed
    // format and one otherwise. A copy split by rows states its range in texels,
    // so this is what those numbers are multiples of.
    block_height: u32,
    level_count: usize,
    level_storage: [max_levels]Level,

    // Ascending mip index: entry zero is the base level. The file stores them in
    // the opposite order, which parse checks and undoes.
    pub fn levels(self: *const File) []const Level {
        return self.level_storage[0..self.level_count];
    }
};

pub fn isKtx2(bytes: []const u8) bool {
    return bytes.len >= identifier.len and
        std.mem.eql(u8, bytes[0..identifier.len], &identifier);
}

pub fn parse(bytes: []const u8) ParseError!File {
    if (bytes.len < header_size or !isKtx2(bytes)) return error.InvalidIdentifier;
    const file_size: u64 = bytes.len;

    const supported = formatFromCode(readU32(bytes, 12)) orelse
        return error.UnsupportedFormat;
    if (readU32(bytes, 16) != supported.type_size) return error.UnsupportedFormat;

    const pixel_width = readU32(bytes, 20);
    const pixel_height = readU32(bytes, 24);
    const pixel_depth = readU32(bytes, 28);
    const layer_count = readU32(bytes, 32);
    const face_count = readU32(bytes, 36);
    const level_count = readU32(bytes, 40);
    const supercompression = readU32(bytes, 44);

    // KTX File Format Specification v2, header: pixelDepth zero means a 2D
    // image and layerCount zero means non-array, and both fields distinguish a
    // zero from a one.
    if (pixel_depth != 0 or layer_count != 0) return error.UnsupportedLayout;
    const kind: Kind = switch (face_count) {
        1 => .texture_2d,
        cube_faces => .cube,
        else => return error.UnsupportedLayout,
    };
    if (supercompression != 0) return error.UnsupportedSupercompression;
    if (pixel_width == 0 or pixel_height == 0) return error.InvalidDimensions;
    // Vulkan specification, VkImageCreateInfo: a cube-compatible image has equal
    // width and height. A file that is not square cannot become one, so it is
    // refused here rather than at image creation.
    if (kind == .cube and pixel_width != pixel_height) return error.InvalidDimensions;
    if (level_count == 0) return error.MissingMipChain;
    if (level_count > max_levels) return error.TooManyLevels;

    const full_chain = std.math.log2_int(u32, @max(pixel_width, pixel_height)) + 1;
    switch (kind) {
        // A partial chain is rejected rather than uploaded. Sampling a minified
        // texture without its small levels aliases, and this reader exists for
        // artifacts whose chain is baked offline.
        .texture_2d => if (level_count != full_chain) return error.InvalidMipChain,
        // An environment's levels are roughness steps, sampled at a level the
        // shader computes rather than at one minification picks, so a short
        // chain is a prefilter's choice and not a missing tail. Khronos'
        // lambertian irradiance map ships as a single level at full extent.
        .cube => if (level_count > full_chain) return error.InvalidMipChain,
    }

    const index_end = header_size + @as(u64, level_count) * level_index_entry_size;
    if (file_size < index_end) return error.Truncated;
    const index_range = Range{ .start = 0, .end = index_end };

    const dfd = try optionalRange(file_size, readU32(bytes, 48), readU32(bytes, 52));
    const kvd = try optionalRange(file_size, readU32(bytes, 56), readU32(bytes, 60));
    const sgd = try optionalRange(file_size, readU64(bytes, 64), readU64(bytes, 72));

    // KTX File Format Specification v2 requires the data format descriptor and
    // makes the other two sections optional.
    const dfd_range = dfd orelse return error.InvalidDataFormat;
    if (dfd_range.overlaps(index_range)) return error.InvalidDataFormat;
    try validateDataFormat(bytes, dfd_range, supported);

    if (kvd) |range| {
        if (range.overlaps(index_range) or range.overlaps(dfd_range))
            return error.OverlappingData;
    }
    if (sgd) |range| {
        if (range.overlaps(index_range) or range.overlaps(dfd_range))
            return error.OverlappingData;
        if (kvd) |other| if (range.overlaps(other)) return error.OverlappingData;
    }

    const alignment = levelAlignment(supported);
    var file: File = .{
        .format = supported.format,
        .kind = kind,
        .face_count = face_count,
        .width = pixel_width,
        .height = pixel_height,
        .level_alignment = alignment,
        .block_height = supported.block_height,
        .level_count = level_count,
        .level_storage = undefined,
    };

    var mip_width = pixel_width;
    var mip_height = pixel_height;
    for (file.level_storage[0..level_count], 0..) |*level, index| {
        const entry = header_size + @as(u64, index) * level_index_entry_size;
        const byte_offset = readU64(bytes, @intCast(entry));
        const byte_length = readU64(bytes, @intCast(entry + 8));
        const uncompressed_length = readU64(bytes, @intCast(entry + 16));

        // Without supercompression the two lengths describe the same bytes, and
        // the length is fixed by the block count times the face count: a file
        // that disagrees is not the image it claims to be.
        const blocks_x = (@as(u64, mip_width) + supported.block_width - 1) / supported.block_width;
        const blocks_y = (@as(u64, mip_height) + supported.block_height - 1) / supported.block_height;
        const face_byte_length = blocks_x * blocks_y * supported.block_bytes;
        if (byte_length != face_byte_length * face_count or
            uncompressed_length != byte_length)
            return error.InvalidLevel;
        if (byte_offset % alignment != 0) return error.InvalidLevel;

        const range = try dataRange(file_size, byte_offset, byte_length);
        if (range.overlaps(index_range) or range.overlaps(dfd_range))
            return error.OverlappingData;
        if (kvd) |other| if (range.overlaps(other)) return error.OverlappingData;
        if (sgd) |other| if (range.overlaps(other)) return error.OverlappingData;
        for (file.level_storage[0..index]) |previous| {
            const previous_range = Range{
                .start = previous.byte_offset,
                .end = previous.byte_offset + previous.byte_length,
            };
            if (range.overlaps(previous_range)) return error.OverlappingData;
        }

        level.* = .{
            .byte_offset = byte_offset,
            .byte_length = byte_length,
            .face_byte_length = face_byte_length,
            .row_bytes = blocks_x * supported.block_bytes,
            .width = mip_width,
            .height = mip_height,
        };
        mip_width = @max(mip_width >> 1, 1);
        mip_height = @max(mip_height >> 1, 1);
    }

    try validatePacking(file.levels(), alignment, file_size, index_range, dfd_range, kvd, sgd);
    return file;
}

// KTX File Format Specification v2, level index: each level's data is aligned to
// the least common multiple of the texel block size and 4. For every 4x4 block
// compressed format that is the block's own 16 bytes; for the eight-byte texel
// of R16G16B16A16_SFLOAT it is 8, and a reader carrying a constant 16 refuses
// such a file on an alignment error that describes nothing real.
fn levelAlignment(supported: Supported) u64 {
    const four: u64 = 4;
    return supported.block_bytes / std.math.gcd(supported.block_bytes, four) * four;
}

// KTX File Format Specification v2, level index: level data is stored smallest
// level first, packed without gaps, each level aligned. Checking the exact
// layout rather than only the bounds is what refuses a file whose index points
// into plausible but wrong bytes.
fn validatePacking(
    levels: []const Level,
    alignment: u64,
    file_size: u64,
    index_range: Range,
    dfd: Range,
    kvd: ?Range,
    sgd: ?Range,
) ParseError!void {
    var offset = @max(index_range.end, dfd.end);
    if (kvd) |range| offset = @max(offset, range.end);
    if (sgd) |range| offset = @max(offset, range.end);
    offset = std.mem.alignForward(u64, offset, alignment);

    var index = levels.len;
    while (index > 0) {
        index -= 1;
        if (levels[index].byte_offset != offset) return error.InvalidLevel;
        offset += levels[index].byte_length;
    }
    if (offset != file_size) return error.InvalidLevel;
}

const Range = struct {
    start: u64,
    end: u64,

    fn overlaps(a: Range, b: Range) bool {
        return a.start < b.end and b.start < a.end;
    }
};

fn optionalRange(file_size: u64, offset: u64, length: u64) ParseError!?Range {
    if (length == 0) return null;
    return try dataRange(file_size, offset, length);
}

fn dataRange(file_size: u64, offset: u64, length: u64) ParseError!Range {
    if (offset > file_size or length > file_size - offset) return error.Truncated;
    return .{ .start = offset, .end = offset + length };
}

fn formatFromCode(code: u32) ?Supported {
    // The level sizing and the offset alignment are written once for every row
    // of the table, so a row whose geometry is degenerate would produce a
    // division by zero or a length of zero rather than a rejection. It sits here
    // rather than at file scope because a comptime block at file scope is never
    // analysed.
    comptime {
        for (supported_formats) |supported| {
            std.debug.assert(supported.block_width > 0);
            std.debug.assert(supported.block_height > 0);
            std.debug.assert(supported.block_bytes > 0);
            std.debug.assert(supported.type_size > 0);
            // A block-compressed format states typeSize 1 whatever its block
            // holds; an uncompressed one has whole channels in every texel.
            const compressed = supported.block_width > 1 or supported.block_height > 1;
            if (compressed) {
                std.debug.assert(supported.type_size == 1);
            } else {
                std.debug.assert(supported.block_bytes % supported.type_size == 0);
            }
        }
    }
    for (supported_formats) |candidate| {
        if (code == @intFromEnum(candidate.format)) return candidate;
    }
    return null;
}

// Khronos Data Format Specification v1.3, basic descriptor block. The fields
// checked here are the ones that would let a file claim one format and carry
// another: a mismatch is a wrongly decoded texture rather than a crash, which is
// why it is checked at all. It is also the only check that separates two formats
// of equal size, which the level lengths cannot do.
//
// A format whose table row has no descriptor skips the content checks. The
// section is still required to exist and still may not overlap anything, so a
// file cannot omit it.
const dfd_minimum_size: u64 = 28;
const dfd_basic_descriptor_type: u16 = 0;
const dfd_version: u16 = 2;
const dfd_primaries_bt709: u8 = 1;
const dfd_flags_alpha_straight: u8 = 0;

fn validateDataFormat(bytes: []const u8, range: Range, supported: Supported) ParseError!void {
    const length = range.end - range.start;
    if (length < dfd_minimum_size) return error.InvalidDataFormat;
    const expected = supported.descriptor orelse return;

    const start: usize = @intCast(range.start);
    const dfd = bytes[start..][0..@intCast(length)];

    // KTX File Format Specification v2: dfdTotalSize counts itself, so the
    // descriptor block that follows is four bytes shorter.
    const descriptor_size = std.mem.readInt(u16, dfd[10..12], .little);
    if (std.mem.readInt(u32, dfd[0..4], .little) != length or
        std.mem.readInt(u16, dfd[4..6], .little) != 0 or
        std.mem.readInt(u16, dfd[6..8], .little) != dfd_basic_descriptor_type or
        std.mem.readInt(u16, dfd[8..10], .little) != dfd_version or
        @as(u64, descriptor_size) + 4 != length)
        return error.InvalidDataFormat;

    // Khronos Data Format Specification v1.3, basic descriptor block:
    // texelBlockDimension is stored as the dimension minus one, and the two
    // unused dimensions are zero.
    const block_dimensions = [4]u8{
        @intCast(supported.block_width - 1),
        @intCast(supported.block_height - 1),
        0,
        0,
    };
    if (dfd[12] != expected.colour_model or
        dfd[13] != dfd_primaries_bt709 or
        dfd[14] != expected.transfer_function or
        dfd[15] != dfd_flags_alpha_straight or
        !std.mem.eql(u8, dfd[16..20], &block_dimensions) or
        dfd[20] != supported.block_bytes)
        return error.InvalidDataFormat;
}

// KTX File Format Specification v2: every field is little-endian. Version 2
// dropped the endianness marker that version 1 carried.
fn readU32(bytes: []const u8, offset: usize) u32 {
    return std.mem.readInt(u32, bytes[offset..][0..4], .little);
}

fn readU64(bytes: []const u8, offset: usize) u64 {
    return std.mem.readInt(u64, bytes[offset..][0..8], .little);
}
