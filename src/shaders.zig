const std = @import("std");

// The SPIR-V this module ships, and the only place that embeds any.
//
// One binary per Slang source file, carrying every entry point that file
// declares. A stage is selected by entry point name at pipeline creation, which
// is why the names are here beside the words rather than spelled at each call
// site.
//
// A project supplying its own shading is not in this table. It links against
// the interfaces in `assets/shaders/` and hands the result to
// `pipeline.createModule`, which takes SPIR-V words from anywhere.

// `@embedFile` yields bytes, and SPIR-V is words. The array is copied into an
// aligned constant so the reinterpretation below is valid rather than merely
// likely: an embedded file has no alignment of its own.
fn words(comptime bytes: anytype) []const u32 {
    const aligned: [bytes.len]u8 align(@alignOf(u32)) = bytes;
    const count = aligned.len / @sizeOf(u32);
    // Vulkan specification, VkShaderModuleCreateInfo: codeSize is a multiple of
    // four. Dividing would otherwise drop a partial word and hand the driver a
    // module shorter than the file, which reads as a corrupt shader rather than
    // as a truncated build output.
    comptime std.debug.assert(count * @sizeOf(u32) == aligned.len);
    return @as([*]const u32, @ptrCast(&aligned))[0..count];
}

comptime {
    // Vulkan specification, VkShaderModuleCreateInfo: codeSize is a multiple of
    // four. A truncated file would otherwise reach the driver as a short module.
    for (all) |module| std.debug.assert(module.spirv.len > 0);
}

// SPIR-V words together with the name to enter them by. Distinct from
// `pipeline.Stage`, which names a module the device has already created.
pub const EntryPoint = struct {
    spirv: []const u32,
    name: [*:0]const u8,
};

pub const Module = struct {
    name: []const u8,
    spirv: []const u32,
    vertex_entry: [*:0]const u8 = "vertexMain",
    // The skinned vertex variant, where a module has one. Same module and same
    // bindings; a pipeline selects it by name.
    skinned_vertex_entry: [*:0]const u8 = "skinnedVertexMain",
    fragment_entry: [*:0]const u8 = "fragmentMain",

    pub fn vertex(self: Module) EntryPoint {
        return .{ .spirv = self.spirv, .name = self.vertex_entry };
    }

    pub fn fragment(self: Module) EntryPoint {
        return .{ .spirv = self.spirv, .name = self.fragment_entry };
    }
};

// The compiler's reflection output for a module, as emitted beside its words.
// Only the checks read it; nothing on a frame path parses JSON.
pub const Reflection = struct {
    name: []const u8,
    json: []const u8,
};

pub const reflection = [_]Reflection{
    .{ .name = "fullscreen", .json = @embedFile("fullscreen_reflection") },
    .{ .name = "scene", .json = @embedFile("scene_reflection") },
};

// The main pass: one instanced mesh, transformed and textured.
pub const scene: Module = .{
    .name = "scene",
    .spirv = words(@embedFile("scene").*),
};

// The post pass: a screen-covering triangle that samples the HDR target.
pub const fullscreen: Module = .{
    .name = "fullscreen",
    .spirv = words(@embedFile("fullscreen").*),
};

// Everything above, for the checks that walk the set rather than name one.
pub const all = [_]Module{ fullscreen, scene };
