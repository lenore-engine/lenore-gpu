const std = @import("std");
const gpu = @import("lenore-gpu");

const testing = std.testing;

// SPIR-V specification, 3.1 Magic Number: 0x07230203, and the module is a
// stream of words rather than bytes, so reading it back in the host's word
// order is what this confirms.
const spirv_magic: u32 = 0x07230203;

test "every embedded module is SPIR-V and not something else" {
    // The build step names the output; a rename or a slangc failure that still
    // produced a file would reach the driver as a module rather than as a build
    // error. The magic number is what tells the two apart offline.
    try testing.expect(gpu.Shaders.all.len > 0);
    for (gpu.Shaders.all) |module| {
        try testing.expect(module.spirv.len >= 5);
        try testing.expectEqual(spirv_magic, module.spirv[0]);
    }
}

test "a module's two stages differ only in the entry point they name" {
    const fullscreen = gpu.Shaders.fullscreen;
    const vertex = fullscreen.vertex() orelse return error.TestExpectedVertexEntry;
    const fragment = fullscreen.fragment() orelse return error.TestExpectedFragmentEntry;

    // One binary, two entry points. Distinct words would mean the build went
    // back to a module per stage without this side noticing.
    try testing.expectEqual(fullscreen.spirv.ptr, vertex.spirv.ptr);
    try testing.expectEqual(fullscreen.spirv.ptr, fragment.spirv.ptr);

    const vertex_name = std.mem.span(vertex.name);
    const fragment_name = std.mem.span(fragment.name);
    try testing.expect(!std.mem.eql(u8, vertex_name, fragment_name));
    try testing.expectEqualStrings("vertexMain", vertex_name);
    try testing.expectEqualStrings("fragmentMain", fragment_name);
}

test "the instruction stream ends exactly where the slice does" {
    // SPIR-V specification, 2.3 Physical Layout: words 0 to 4 are the header
    // and word 5 begins the instruction stream, where each instruction's first
    // word carries its own length in the 16 high-order bits.
    //
    // Walking it is what pins the word count. The magic number alone does not:
    // a slice four times too long starts with the same word, and hands the
    // driver a codeSize that reads past the embedded bytes.
    const header_words = 5;

    for (gpu.Shaders.all) |module| {
        var index: usize = header_words;
        while (index < module.spirv.len) {
            const length = module.spirv[index] >> 16;
            try testing.expect(length > 0);
            index += length;
        }
        try testing.expectEqual(module.spirv.len, index);
    }
}

test "the words are aligned for the driver to read" {
    // Vulkan specification, VkShaderModuleCreateInfo: pCode is a pointer to
    // four-byte-aligned words. `@embedFile` gives bytes with no alignment of
    // their own, so the copy that makes this true is the load-bearing part.
    for (gpu.Shaders.all) |module| {
        try testing.expect(std.mem.isAligned(@intFromPtr(module.spirv.ptr), @alignOf(u32)));
    }
}
