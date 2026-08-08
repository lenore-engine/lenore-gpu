const std = @import("std");
const gpu = @import("lenore-gpu");

const testing = std.testing;

// A 4x4 BC7 image: three levels of one 16-byte block each. The whole file is
// laid out by hand so a test can corrupt one field and see which check catches
// it. Sizes are stated rather than computed so a mistake in the builder shows up
// as a failing valid-file test instead of hiding inside every other case.
//
// 0    identifier and header          80
// 80   level index, three entries     72
// 152  data format descriptor         28
// 180  padding to the block alignment 12
// 192  level 2, 1x1                   16
// 208  level 1, 2x2                   16
// 224  level 0, 4x4                   16
// 240  end of file
const file_size = 240;
const index_offset = 80;
const dfd_offset = 152;
const dfd_length = 28;
const level_bytes = 16;
const smallest_level_offset = 192;

const bc7_srgb_format: u32 = 146;
const rgba16_sfloat_format: u32 = 97;

const File = [file_size]u8;

fn validFile() File {
    var bytes: File = @splat(0);

    writeIdentifier(&bytes);
    put32(&bytes, 12, bc7_srgb_format);
    put32(&bytes, 16, 1); // typeSize
    put32(&bytes, 20, 4); // pixelWidth
    put32(&bytes, 24, 4); // pixelHeight
    put32(&bytes, 28, 0); // pixelDepth, zero meaning 2D
    put32(&bytes, 32, 0); // layerCount, zero meaning non-array
    put32(&bytes, 36, 1); // faceCount
    put32(&bytes, 40, 3); // levelCount
    put32(&bytes, 44, 0); // supercompressionScheme
    put32(&bytes, 48, dfd_offset);
    put32(&bytes, 52, dfd_length);
    put32(&bytes, 56, 0); // no key/value data
    put32(&bytes, 60, 0);
    put64(&bytes, 64, 0); // no supercompression global data
    put64(&bytes, 72, 0);

    // Ascending mip index in the index, descending order in the file: the
    // smallest level is stored first.
    for (0..3) |level| {
        const entry = index_offset + level * 24;
        const offset = smallest_level_offset + (2 - level) * level_bytes;
        put64(&bytes, entry, offset);
        put64(&bytes, entry + 8, level_bytes);
        put64(&bytes, entry + 16, level_bytes);
    }

    put32(&bytes, dfd_offset, dfd_length);
    put16(&bytes, dfd_offset + 4, 0); // vendorId
    put16(&bytes, dfd_offset + 6, 0); // descriptorType
    put16(&bytes, dfd_offset + 8, 2); // versionNumber
    put16(&bytes, dfd_offset + 10, dfd_length - 4);
    bytes[dfd_offset + 12] = 134; // colorModel, BC7
    bytes[dfd_offset + 13] = 1; // colorPrimaries, BT709
    bytes[dfd_offset + 14] = 2; // transferFunction, sRGB
    bytes[dfd_offset + 15] = 0; // flags
    @memcpy(bytes[dfd_offset + 16 ..][0..4], &[4]u8{ 3, 3, 0, 0 });
    bytes[dfd_offset + 20] = 16; // bytesPlane0

    return bytes;
}

// A 2x2 R16G16B16A16_SFLOAT cube map with a complete two-level chain, in the
// shape of a prefiltered environment. Every level holds six faces, so its length
// is six times one face, and the level alignment is the eight-byte texel rather
// than the sixteen a block-compressed file uses.
//
// The descriptor is placed eight bytes past the index so that the first level
// lands on 168: divisible by the eight-byte texel and not by sixteen. A parser
// carrying the block-compressed alignment then fails this fixture instead of
// passing it by coincidence.
//
// 0    identifier and header          80
// 80   level index, two entries       48
// 128  padding                         8
// 136  data format descriptor         28
// 164  padding to the texel alignment  4
// 168  level 1, 1x1, six faces        48
// 216  level 0, 2x2, six faces       192
// 408  end of file
const cube_file_size = 408;
const cube_index_offset = 80;
const cube_dfd_offset = 136;
const cube_smallest_level_offset = 168;
const cube_base_level_offset = 216;

const CubeFile = [cube_file_size]u8;

fn validCubeFile() CubeFile {
    var bytes: CubeFile = @splat(0);

    writeIdentifier(&bytes);
    put32(&bytes, 12, rgba16_sfloat_format);
    put32(&bytes, 16, 2); // typeSize, one 16-bit channel
    put32(&bytes, 20, 2); // pixelWidth
    put32(&bytes, 24, 2); // pixelHeight
    put32(&bytes, 28, 0);
    put32(&bytes, 32, 0);
    put32(&bytes, 36, 6); // faceCount
    put32(&bytes, 40, 2); // levelCount
    put32(&bytes, 44, 0);
    put32(&bytes, 48, cube_dfd_offset);
    put32(&bytes, 52, dfd_length);
    put32(&bytes, 56, 0);
    put32(&bytes, 60, 0);
    put64(&bytes, 64, 0);
    put64(&bytes, 72, 0);

    put64(&bytes, cube_index_offset, cube_base_level_offset);
    put64(&bytes, cube_index_offset + 8, 192);
    put64(&bytes, cube_index_offset + 16, 192);
    put64(&bytes, cube_index_offset + 24, cube_smallest_level_offset);
    put64(&bytes, cube_index_offset + 32, 48);
    put64(&bytes, cube_index_offset + 40, 48);

    writePlaceholderDescriptor(&bytes, cube_dfd_offset);
    return bytes;
}

// The descriptor Khronos' own prefiltered environments carry, byte for byte:
// dfdTotalSize zero, colour model and primaries unspecified, no samples and
// bytesPlane all zero. The parser must accept this file, so the fixture states
// it exactly rather than writing a conformant descriptor the real assets lack.
fn writePlaceholderDescriptor(bytes: []u8, offset: usize) void {
    put32(bytes, offset, 0); // dfdTotalSize, nonconformant and left as written
    put16(bytes, offset + 4, 0); // vendorId
    put16(bytes, offset + 6, 0); // descriptorType
    put16(bytes, offset + 8, 2); // versionNumber
    put16(bytes, offset + 10, 24); // descriptorBlockSize, no samples
    bytes[offset + 12] = 0; // colorModel, unspecified
    bytes[offset + 13] = 0; // colorPrimaries, unspecified
    bytes[offset + 14] = 1; // transferFunction, linear
    bytes[offset + 15] = 0; // flags
}

fn writeIdentifier(bytes: []u8) void {
    const identifier = [12]u8{
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32,
        0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
    };
    @memcpy(bytes[0..12], &identifier);
}

fn put16(bytes: []u8, offset: usize, value: u16) void {
    std.mem.writeInt(u16, bytes[offset..][0..2], value, .little);
}

fn put32(bytes: []u8, offset: usize, value: u32) void {
    std.mem.writeInt(u32, bytes[offset..][0..4], value, .little);
}

fn put64(bytes: []u8, offset: usize, value: u64) void {
    std.mem.writeInt(u64, bytes[offset..][0..8], value, .little);
}

test "a valid file yields an ascending mip chain" {
    const bytes = validFile();
    const file = try gpu.parseKtx2(&bytes);

    try testing.expectEqual(gpu.Ktx2Kind.texture_2d, file.kind);
    try testing.expectEqual(@as(u32, 1), file.face_count);
    try testing.expectEqual(@as(u32, 4), file.width);
    try testing.expectEqual(@as(u32, 4), file.height);
    try testing.expectEqual(@as(usize, 3), file.level_count);

    const levels = file.levels();
    try testing.expectEqual(@as(u32, 4), levels[0].width);
    try testing.expectEqual(@as(u32, 2), levels[1].width);
    try testing.expectEqual(@as(u32, 1), levels[2].width);
    try testing.expectEqual(@as(u32, 1), levels[2].height);

    // Ascending mip index, descending file offset.
    try testing.expectEqual(@as(u64, 224), levels[0].byte_offset);
    try testing.expectEqual(@as(u64, 208), levels[1].byte_offset);
    try testing.expectEqual(@as(u64, 192), levels[2].byte_offset);
    for (levels) |level| {
        try testing.expectEqual(@as(u64, 16), level.byte_length);
        // One face, so the two lengths coincide and a caller that confuses them
        // is not caught here. The cube fixture below is what separates them.
        try testing.expectEqual(@as(u64, 16), level.face_byte_length);
    }
}

test "the identifier gates everything" {
    var bytes = validFile();
    bytes[0] = 0;
    try testing.expectError(error.InvalidIdentifier, gpu.parseKtx2(&bytes));

    try testing.expect(!gpu.isKtx2(bytes[0..4]));
    try testing.expect(gpu.isKtx2(validFile()[0..12]));
    try testing.expectError(error.InvalidIdentifier, gpu.parseKtx2(validFile()[0..40]));
}

test "unsupported formats and container features are refused" {
    {
        var bytes = validFile();
        put32(&bytes, 12, 37); // r8g8b8a8_unorm, in no accepted row
        try testing.expectError(error.UnsupportedFormat, gpu.parseKtx2(&bytes));
    }
    {
        var bytes = validFile();
        put32(&bytes, 16, 4); // typeSize, meaningless for a block format
        try testing.expectError(error.UnsupportedFormat, gpu.parseKtx2(&bytes));
    }
    {
        // The float format's typeSize is its channel size, not one.
        var bytes = validCubeFile();
        put32(&bytes, 16, 1);
        try testing.expectError(error.UnsupportedFormat, gpu.parseKtx2(&bytes));
    }
    {
        var bytes = validFile();
        put32(&bytes, 44, 1); // BasisLZ supercompression
        try testing.expectError(error.UnsupportedSupercompression, gpu.parseKtx2(&bytes));
    }
    for ([_]struct { usize, u32 }{
        .{ 28, 1 }, // pixelDepth: a 3D image
        .{ 32, 2 }, // layerCount: an array
        .{ 36, 2 }, // faceCount: neither a 2D image nor a cube
        .{ 36, 5 }, // faceCount: an incomplete cube
    }) |patch| {
        var bytes = validFile();
        put32(&bytes, patch[0], patch[1]);
        try testing.expectError(error.UnsupportedLayout, gpu.parseKtx2(&bytes));
    }
}

test "the mip chain must be complete and within the cap" {
    {
        var bytes = validFile();
        put32(&bytes, 40, 0);
        try testing.expectError(error.MissingMipChain, gpu.parseKtx2(&bytes));
    }
    {
        var bytes = validFile();
        put32(&bytes, 40, 17);
        try testing.expectError(error.TooManyLevels, gpu.parseKtx2(&bytes));
    }
    {
        // Two levels for a 4x4 image: the 1x1 level is missing.
        var bytes = validFile();
        put32(&bytes, 40, 2);
        try testing.expectError(error.InvalidMipChain, gpu.parseKtx2(&bytes));
    }
    {
        var bytes = validFile();
        put32(&bytes, 20, 0);
        try testing.expectError(error.InvalidDimensions, gpu.parseKtx2(&bytes));
    }
}

test "the data format descriptor must match the format field" {
    {
        var bytes = validFile();
        put32(&bytes, 52, 0); // no descriptor at all
        try testing.expectError(error.InvalidDataFormat, gpu.parseKtx2(&bytes));
    }
    {
        var bytes = validFile();
        bytes[dfd_offset + 12] = 132; // BC5 model in a BC7 file
        try testing.expectError(error.InvalidDataFormat, gpu.parseKtx2(&bytes));
    }
    {
        var bytes = validFile();
        bytes[dfd_offset + 14] = 1; // linear transfer in an sRGB file
        try testing.expectError(error.InvalidDataFormat, gpu.parseKtx2(&bytes));
    }
    {
        var bytes = validFile();
        bytes[dfd_offset + 16] = 4; // a 5x4 texel block
        try testing.expectError(error.InvalidDataFormat, gpu.parseKtx2(&bytes));
    }
    {
        var bytes = validFile();
        bytes[dfd_offset + 20] = 8; // eight bytes per block
        try testing.expectError(error.InvalidDataFormat, gpu.parseKtx2(&bytes));
    }
    {
        var bytes = validFile();
        put16(&bytes, dfd_offset + 10, dfd_length); // block size disagrees with the total
        try testing.expectError(error.InvalidDataFormat, gpu.parseKtx2(&bytes));
    }
}

test "a level whose length is not its block count is refused" {
    var bytes = validFile();
    put64(&bytes, index_offset + 8, 32); // base level claims two blocks
    try testing.expectError(error.InvalidLevel, gpu.parseKtx2(&bytes));
}

test "a level offset must be block aligned and gapless" {
    {
        var bytes = validFile();
        put64(&bytes, index_offset, 225); // base level off the block grid
        try testing.expectError(error.InvalidLevel, gpu.parseKtx2(&bytes));
    }
    {
        // A gap: every level shifted up by one block leaves the file short.
        var bytes = validFile();
        for (0..3) |level| {
            const entry = index_offset + level * 24;
            put64(&bytes, entry, smallest_level_offset + (2 - level) * level_bytes + 16);
        }
        try testing.expectError(error.Truncated, gpu.parseKtx2(&bytes));
    }
    {
        // In range but not where the packing rule puts it.
        var bytes = validFile();
        put64(&bytes, index_offset + 2 * 24, 208);
        put64(&bytes, index_offset + 24, 192);
        try testing.expectError(error.InvalidLevel, gpu.parseKtx2(&bytes));
    }
}

test "level data may not overlap the metadata or another level" {
    {
        var bytes = validFile();
        // Block aligned and inside the descriptor's range, so the overlap check
        // is what has to catch it rather than the alignment check.
        put64(&bytes, index_offset + 2 * 24, 160);
        try testing.expectError(error.OverlappingData, gpu.parseKtx2(&bytes));
    }
    {
        var bytes = validFile();
        put64(&bytes, index_offset + 24, 192); // two levels on the same block
        try testing.expectError(error.OverlappingData, gpu.parseKtx2(&bytes));
    }
}

// The key/value and supercompression-global sections are optional and the
// fixture above declares neither, so the checks that keep them from overlapping
// anything need a file that has them.
test "metadata sections may not overlap the index, the descriptor or a level" {
    // Both sections are declared over the descriptor's own bytes, which is the
    // simplest way to make a well-formed file whose regions collide.
    {
        var bytes = validFile();
        put32(&bytes, 56, dfd_offset);
        put32(&bytes, 60, dfd_length);
        try testing.expectError(error.OverlappingData, gpu.parseKtx2(&bytes));
    }
    {
        var bytes = validFile();
        put64(&bytes, 64, dfd_offset);
        put64(&bytes, 72, dfd_length);
        try testing.expectError(error.OverlappingData, gpu.parseKtx2(&bytes));
    }
    {
        // Key/value data over the level index.
        var bytes = validFile();
        put32(&bytes, 56, index_offset);
        put32(&bytes, 60, 24);
        try testing.expectError(error.OverlappingData, gpu.parseKtx2(&bytes));
    }
    {
        // Supercompression global data over the key/value data.
        var bytes = validFile();
        put32(&bytes, 56, dfd_offset + dfd_length);
        put32(&bytes, 60, 4);
        put64(&bytes, 64, dfd_offset + dfd_length);
        put64(&bytes, 72, 4);
        try testing.expectError(error.OverlappingData, gpu.parseKtx2(&bytes));
    }
    {
        // A section reaching past the end is truncated rather than overlapping.
        var bytes = validFile();
        put32(&bytes, 56, file_size - 4);
        put32(&bytes, 60, 64);
        try testing.expectError(error.Truncated, gpu.parseKtx2(&bytes));
    }
}

// A section that sits in the padding between the descriptor and the first level
// is legal, and the levels must still be found where the packing rule puts them.
test "metadata in the padding does not disturb the level layout" {
    var bytes = validFile();
    put32(&bytes, 56, dfd_offset + dfd_length);
    put32(&bytes, 60, 12);

    const file = try gpu.parseKtx2(&bytes);
    try testing.expectEqual(@as(usize, 3), file.level_count);
    try testing.expectEqual(@as(u64, 224), file.levels()[0].byte_offset);
}

test "a level range outside the file is refused" {
    var bytes = validFile();
    put64(&bytes, index_offset, 240); // starts exactly at the end
    try testing.expectError(error.Truncated, gpu.parseKtx2(&bytes));
}

test "a cube map's level spans six faces" {
    const bytes = validCubeFile();
    const file = try gpu.parseKtx2(&bytes);

    try testing.expectEqual(gpu.Ktx2Kind.cube, file.kind);
    try testing.expectEqual(@as(u32, 6), file.face_count);
    try testing.expectEqual(@as(u32, 2), file.width);
    try testing.expectEqual(@as(u32, 2), file.height);
    try testing.expectEqual(@as(usize, 2), file.level_count);

    const levels = file.levels();
    // 2x2 texels of eight bytes is 32 for one face and 192 for the cube; the
    // two differ by more than a factor, so a caller using the wrong one cannot
    // land on a plausible extent.
    try testing.expectEqual(@as(u64, 192), levels[0].byte_length);
    try testing.expectEqual(@as(u64, 32), levels[0].face_byte_length);
    try testing.expectEqual(@as(u64, 48), levels[1].byte_length);
    try testing.expectEqual(@as(u64, 8), levels[1].face_byte_length);
    try testing.expectEqual(@as(u64, cube_base_level_offset), levels[0].byte_offset);
    try testing.expectEqual(@as(u64, cube_smallest_level_offset), levels[1].byte_offset);
}

// The lambertian irradiance map Khronos publishes is one level at full extent,
// which the 2D rule would reject as a missing tail.
test "a cube map may carry fewer levels than its extent allows" {
    // One index entry ends at 104, the descriptor at 132, and the single level
    // starts at the next multiple of eight.
    var bytes: [328]u8 = @splat(0);
    @memcpy(bytes[0..80], validCubeFile()[0..80]);
    put32(&bytes, 40, 1); // levelCount
    put32(&bytes, 48, 104);
    put32(&bytes, 52, dfd_length);
    put64(&bytes, cube_index_offset, 136);
    put64(&bytes, cube_index_offset + 8, 192);
    put64(&bytes, cube_index_offset + 16, 192);
    writePlaceholderDescriptor(&bytes, 104);

    const file = try gpu.parseKtx2(&bytes);
    try testing.expectEqual(@as(usize, 1), file.level_count);
    try testing.expectEqual(@as(u64, 192), file.levels()[0].byte_length);

    // More levels than the extent can halve into is still a broken file.
    var too_many = validCubeFile();
    put32(&too_many, 40, 3);
    try testing.expectError(error.InvalidMipChain, gpu.parseKtx2(&too_many));
}

test "a cube map must be square" {
    var bytes = validCubeFile();
    put32(&bytes, 24, 4); // pixelHeight, leaving a 2x4 cube
    try testing.expectError(error.InvalidDimensions, gpu.parseKtx2(&bytes));
}

// A block-compressed file aligns its levels to sixteen bytes and an eight-byte
// texel format to eight. Carrying one constant for both silently refuses every
// real environment, so the fixture pins the smaller one.
test "an uncompressed level aligns to its texel, not to a block" {
    const file = try gpu.parseKtx2(&validCubeFile());
    try testing.expectEqual(@as(u64, 168), file.levels()[1].byte_offset);
    try testing.expectEqual(@as(u64, 8), file.levels()[1].byte_offset % 16);

    var misaligned = validCubeFile();
    put64(&misaligned, cube_index_offset + 24, 172);
    try testing.expectError(error.InvalidLevel, gpu.parseKtx2(&misaligned));
}

// The placeholder descriptor is accepted only where the table says the format
// has none. A block-compressed file carrying the same bytes is still refused,
// so the relaxation cannot leak into the path that has a real check.
test "the placeholder descriptor does not excuse a block-compressed file" {
    var bytes = validFile();
    writePlaceholderDescriptor(&bytes, dfd_offset);
    try testing.expectError(error.InvalidDataFormat, gpu.parseKtx2(&bytes));
}

test "a cube map's level length counts every face" {
    var bytes = validCubeFile();
    put64(&bytes, cube_index_offset + 8, 32); // the base level as if it held one face
    try testing.expectError(error.InvalidLevel, gpu.parseKtx2(&bytes));
}

// An uploader reserves the whole chain as one block and computes each level's
// offset relative to it, so the block has to be reserved at this alignment or
// every level inside it is misaligned by the same amount.
test "the file reports the alignment its own levels were placed on" {
    try testing.expectEqual(@as(u64, 16), (try gpu.parseKtx2(&validFile())).level_alignment);
    try testing.expectEqual(@as(u64, 8), (try gpu.parseKtx2(&validCubeFile())).level_alignment);
}
