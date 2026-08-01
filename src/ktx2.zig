const std = @import("std");
const vk = @import("vulkan");

// A narrow KTX2 reader for block-compressed texture artifacts, safe against
// hostile input: every field is validated before anything is created, and the
// parser touches no allocator and no Vulkan object.
//
// Accepted files are 2D, single-face, non-array, uncompressed-container BC
// images carrying a complete baked mip chain. Layout from the KTX File Format
// Specification v2: a fixed 80-byte header, then one 24-byte level index entry
// per level, all little-endian.

const identifier = [12]u8{ 0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A };
const header_size: u64 = 80;
const level_index_entry_size: u64 = 24;

// The block geometry every accepted format shares. It is stated once here and
// asserted against the table below, so a format whose blocks differ cannot be
// added without the arithmetic that assumes these being revisited.
const block_width: u32 = 4;
const block_height: u32 = 4;
const block_bytes: u64 = 16;

// The same cap the image type enforces on a mip chain, which describes extents
// up to 32768 texels. Holding it here keeps parsing allocation-free.
pub const max_levels: usize = 16;

// Khronos Data Format Specification v1.3, table "Color models": the block
// compressed models are 128 plus the BC index, so BC5 is 132 and BC7 is 134.
// Transfer functions in the same document: KHR_DF_TRANSFER_LINEAR is 1 and
// KHR_DF_TRANSFER_SRGB is 2.
const Supported = struct {
    format: vk.Format,
    colour_model: u8,
    transfer_function: u8,
    // Stated per format rather than assumed, so the assert below can hold the
    // arithmetic in this file to what the table actually contains.
    block_width: u32,
    block_height: u32,
    block_bytes: u64,
};

// Adding a format means adding its descriptor values in the same row, so a
// format can never be accepted without the data-format check that proves the
// file really holds it.
const supported_formats = [_]Supported{
    .{
        .format = .bc5_unorm_block,
        .colour_model = 132,
        .transfer_function = 1,
        .block_width = 4,
        .block_height = 4,
        .block_bytes = 16,
    },
    .{
        .format = .bc7_unorm_block,
        .colour_model = 134,
        .transfer_function = 1,
        .block_width = 4,
        .block_height = 4,
        .block_bytes = 16,
    },
    .{
        .format = .bc7_srgb_block,
        .colour_model = 134,
        .transfer_function = 2,
        .block_width = 4,
        .block_height = 4,
        .block_bytes = 16,
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
    byte_length: u64,
    width: u32,
    height: u32,
};

// The parse result is a value. The level cap makes the chain a fixed array, so
// there is nothing to free and nothing to outlive the call.
pub const File = struct {
    format: vk.Format,
    width: u32,
    height: u32,
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
    // KTX File Format Specification v2, header: typeSize is 1 for a
    // block-compressed format, so anything else contradicts the format field.
    if (readU32(bytes, 16) != 1) return error.UnsupportedFormat;

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
    if (pixel_depth != 0 or layer_count != 0 or face_count != 1)
        return error.UnsupportedLayout;
    if (supercompression != 0) return error.UnsupportedSupercompression;
    if (pixel_width == 0 or pixel_height == 0) return error.InvalidDimensions;
    if (level_count == 0) return error.MissingMipChain;
    if (level_count > max_levels) return error.TooManyLevels;

    // A partial chain is rejected rather than uploaded. Sampling a minified
    // texture without its small levels aliases, and this reader exists for
    // artifacts whose chain is baked offline.
    const expected_levels = std.math.log2_int(u32, @max(pixel_width, pixel_height)) + 1;
    if (level_count != expected_levels) return error.InvalidMipChain;

    const index_end = header_size + @as(u64, level_count) * level_index_entry_size;
    if (file_size < index_end) return error.Truncated;
    const index_range = Range{ .start = 0, .end = index_end };

    const dfd = try optionalRange(file_size, readU32(bytes, 48), readU32(bytes, 52));
    const kvd = try optionalRange(file_size, readU32(bytes, 56), readU32(bytes, 60));
    const sgd = try optionalRange(file_size, readU64(bytes, 64), readU64(bytes, 72));

    // KTX File Format Specification v2 requires the data format descriptor and
    // makes the other two sections optional. It is also the only one that proves
    // the payload matches the format field.
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

    var file: File = .{
        .format = supported.format,
        .width = pixel_width,
        .height = pixel_height,
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
        // the length is fixed by the block count: a file that disagrees is not
        // the image it claims to be.
        const blocks_x = (@as(u64, mip_width) + block_width - 1) / block_width;
        const blocks_y = (@as(u64, mip_height) + block_height - 1) / block_height;
        if (byte_length != blocks_x * blocks_y * block_bytes or
            uncompressed_length != byte_length)
            return error.InvalidLevel;
        if (byte_offset % block_bytes != 0) return error.InvalidLevel;

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
            .width = mip_width,
            .height = mip_height,
        };
        mip_width = @max(mip_width >> 1, 1);
        mip_height = @max(mip_height >> 1, 1);
    }

    try validatePacking(file.levels(), file_size, index_range, dfd_range, kvd, sgd);
    return file;
}

// KTX File Format Specification v2, level index: level data is stored smallest
// level first, packed without gaps, each level aligned. Checking the exact
// layout rather than only the bounds is what refuses a file whose index points
// into plausible but wrong bytes.
fn validatePacking(
    levels: []const Level,
    file_size: u64,
    index_range: Range,
    dfd: Range,
    kvd: ?Range,
    sgd: ?Range,
) ParseError!void {
    var offset = @max(index_range.end, dfd.end);
    if (kvd) |range| offset = @max(offset, range.end);
    if (sgd) |range| offset = @max(offset, range.end);
    offset = std.mem.alignForward(u64, offset, block_bytes);

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
    // The level sizing, the offset alignment and the descriptor check all assume
    // one block geometry. A format whose blocks differ cannot be added without
    // that arithmetic being revisited, and this is what stops it being added
    // quietly. It sits here rather than at file scope because a comptime block
    // at file scope is never analysed.
    comptime {
        for (supported_formats) |supported| {
            std.debug.assert(supported.block_width == block_width);
            std.debug.assert(supported.block_height == block_height);
            std.debug.assert(supported.block_bytes == block_bytes);
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
// why it is checked at all.
const dfd_minimum_size: u64 = 28;
const dfd_basic_descriptor_type: u16 = 0;
const dfd_version: u16 = 2;
const dfd_primaries_bt709: u8 = 1;
const dfd_flags_alpha_straight: u8 = 0;
// Khronos Data Format Specification v1.3, basic descriptor block:
// texelBlockDimension is stored as the dimension minus one, so a 4x4 block is
// three and three, and the two unused dimensions are zero.
const dfd_block_dimensions = [4]u8{ 3, 3, 0, 0 };

fn validateDataFormat(bytes: []const u8, range: Range, supported: Supported) ParseError!void {
    const length = range.end - range.start;
    if (length < dfd_minimum_size) return error.InvalidDataFormat;
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

    if (dfd[12] != supported.colour_model or
        dfd[13] != dfd_primaries_bt709 or
        dfd[14] != supported.transfer_function or
        dfd[15] != dfd_flags_alpha_straight or
        !std.mem.eql(u8, dfd[16..20], &dfd_block_dimensions) or
        dfd[20] != block_bytes)
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
