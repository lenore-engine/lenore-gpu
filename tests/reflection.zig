const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;

// The compiler's account of a shader's layout, held against the hand-written
// mirror. Neither side is generated from the other, so this is what makes a
// silent disagreement between them a failed build.
//
// Only the parts the checks read are named. Slang's reflection carries far more
// than this, and a full model of it would rot against the compiler rather than
// against our shaders.
const Reflection = struct {
    parameters: []const Parameter = &.{},
    entryPoints: []const EntryPoint = &.{},

    const Parameter = struct {
        name: []const u8 = "",
        // Present on an entry point's varying parameters, absent on a binding.
        semanticName: []const u8 = "",
        binding: Binding = .{},
        type: Type = .{},
    };

    // One `[shader(...)]` function. A file emits every one of them into a
    // single module, and a pipeline selects among them by name, so the names
    // and their stages are part of the layout this holds the mirror against.
    const EntryPoint = struct {
        name: []const u8 = "",
        stage: []const u8 = "",
        parameters: []const Parameter = &.{},
        // A compute entry point's workgroup size, which Slang reports as the
        // three numbers behind `[numthreads]`. Empty on the other stages.
        threadGroupSize: []const u32 = &.{},
    };

    const Binding = struct {
        kind: []const u8 = "",
        // Absent means set zero, which is how Slang writes the default space.
        space: u32 = 0,
        index: u32 = 0,
        offset: u32 = 0,
        size: u32 = 0,
        // The distance between two elements of an array field. Zero on
        // anything that is not one.
        elementStride: u32 = 0,
        // How many consecutive locations a varying parameter occupies. One per
        // field of the struct behind it.
        count: u32 = 0,
    };

    const Type = struct {
        kind: []const u8 = "",
        // A structured buffer's element.
        resultType: ?*const Type = null,
        name: []const u8 = "",
        // How a `resource` is shaped. A structured buffer and a texture are
        // both resources and take different descriptor types.
        baseShape: []const u8 = "",
        fields: []const Parameter = &.{},
        sizes: []const Size = &.{},
        // A constant buffer's contents are one level down, in the struct it
        // holds, rather than on the parameter itself. An array's element sits
        // here too.
        elementType: ?*const Type = null,
        elementCount: u32 = 0,
        // Present on `scalar` only. A vector, matrix or array names it one
        // level down, through `elementType`.
        scalarType: []const u8 = "",
    };

    const Size = struct {
        kind: []const u8 = "",
        value: u32 = 0,
        alignment: u32 = 0,
    };
};

fn block(parameter: Reflection.Parameter) !*const Reflection.Type {
    return parameter.type.elementType orelse error.NotABlock;
}

// The compiler's own statement of the block's length, rather than the last
// field's end: trailing padding belongs to the block and no field reports it.
fn uniformSize(contents: *const Reflection.Type) !u32 {
    for (contents.sizes) |size| {
        if (std.mem.eql(u8, size.kind, "uniform")) return size.value;
    }
    return error.NoUniformSize;
}

fn parse(arena: std.mem.Allocator, json: []const u8) !Reflection {
    return std.json.parseFromSliceLeaky(Reflection, arena, json, .{
        .ignore_unknown_fields = true,
    });
}

fn find(parameters: []const Reflection.Parameter, name: []const u8) ?Reflection.Parameter {
    for (parameters) |parameter| {
        if (std.mem.eql(u8, parameter.name, name)) return parameter;
    }
    return null;
}

fn reflectionFor(name: []const u8) []const u8 {
    for (gpu.Shaders.reflection) |entry| {
        if (std.mem.eql(u8, entry.name, name)) return entry.json;
    }
    unreachable;
}

// What a reflected type is made of, past however many vectors, matrices and
// arrays wrap it.
fn scalarTypeOf(reflected: *const Reflection.Type) ![]const u8 {
    var current = reflected;
    while (current.scalarType.len == 0) {
        current = current.elementType orelse return error.NotAScalarType;
    }
    return current.scalarType;
}

// The same, for the mirror's field. An enum answers for its tag, which is what
// the shader reads it as.
//
// Null where the field bottoms out in a struct rather than a number. Those are
// checked by mirroring the struct itself, which is a whole field list rather
// than one name.
fn mirroredScalarName(comptime T: type) !?[]const u8 {
    return switch (@typeInfo(T)) {
        .float => |float| switch (float.bits) {
            32 => "float32",
            else => error.UnmirroredScalar,
        },
        .int => |int| switch (int.bits) {
            32 => if (int.signedness == .unsigned) "uint32" else "int32",
            else => error.UnmirroredScalar,
        },
        .@"enum" => |tag| mirroredScalarName(tag.tag_type),
        .array => |array| mirroredScalarName(array.child),
        .vector => |vector| mirroredScalarName(vector.child),
        .@"struct" => null,
        else => error.UnmirroredScalar,
    };
}

// A hand-written mirror against the compiler's account of the type it mirrors.
//
// The mirror's own fields are the list, so nothing restates them. They are
// matched by name: renaming one on either side fails as a missing field, where
// matching by position would shift every offset after it and still pass.
//
// Length as well as position, because a field narrower than the shader's sits at
// the right offset and the alignment that follows absorbs the difference, so the
// total does not move either and the shader reads the missing lanes out of
// padding. And the total and the field count, because a field the mirror does
// not have would otherwise sit inside the same bytes unnoticed.
fn expectMirrors(comptime T: type, contents: *const Reflection.Type) !void {
    inline for (@typeInfo(T).@"struct".fields) |mirrored| {
        const field = find(contents.fields, mirrored.name) orelse return error.MissingField;
        try testing.expectEqual(
            @as(u32, @intCast(@offsetOf(T, mirrored.name))),
            field.binding.offset,
        );
        try testing.expectEqual(
            @as(u32, @intCast(@sizeOf(mirrored.type))),
            field.binding.size,
        );

        // What it is made of, not only how much room it takes. A float where
        // the mirror has an integer occupies the same four bytes at the same
        // offset and reads the same word as a different number, which is how
        // the reference's float type tag went unnoticed.
        if (try mirroredScalarName(mirrored.type)) |named| {
            try testing.expectEqualStrings(named, try scalarTypeOf(&field.type));
        }
    }
    try testing.expectEqual(@as(u32, @intCast(@sizeOf(T))), try uniformSize(contents));
    try testing.expectEqual(@typeInfo(T).@"struct".fields.len, contents.fields.len);
}

test "the camera block is what the shader reads, field by field" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const parsed = try parse(arena.allocator(), reflectionFor("scene"));
    const camera = find(parsed.parameters, "camera") orelse return error.MissingCameraBlock;
    try expectMirrors(gpu.CameraUniform, try block(camera));
}

test "the lights block is what the shader reads, field by field" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const parsed = try parse(arena.allocator(), reflectionFor("scene"));
    const lights = find(parsed.parameters, "lights") orelse return error.MissingLightsBlock;
    try expectMirrors(gpu.LightsUniform, try block(lights));
}

test "the post push constants are what the fullscreen shader reads" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const parsed = try parse(arena.allocator(), reflectionFor("fullscreen"));
    const post = find(parsed.parameters, "post") orelse return error.MissingPostPushConstants;
    try testing.expectEqualStrings("pushConstantBuffer", post.binding.kind);
    try expectMirrors(gpu.PostPass.PushConstants, try block(post));

    var push_count: usize = 0;
    for (parsed.parameters) |parameter| {
        if (std.mem.eql(u8, parameter.binding.kind, "pushConstantBuffer")) push_count += 1;
    }
    try testing.expectEqual(@as(usize, 1), push_count);
}

test "a light is the element the shader's array steps through" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const parsed = try parse(arena.allocator(), reflectionFor("scene"));
    const lights = find(parsed.parameters, "lights") orelse return error.MissingLightsBlock;
    const array = find((try block(lights)).fields, "lights") orelse return error.MissingLightArray;

    // The stride is what the block's own length cannot catch: a record shorter
    // than the shader's element still fits inside the array's thousand and
    // twenty-four bytes, and every light after the first is then read from the
    // wrong offset.
    try testing.expectEqual(
        @as(u32, @intCast(@sizeOf(gpu.LightUniform))),
        array.binding.elementStride,
    );
    try testing.expectEqual(@as(u32, gpu.max_lights), array.type.elementCount);

    const element = array.type.elementType orelse return error.NotAnArray;
    try expectMirrors(gpu.LightUniform, element);
}

test "the descriptor sets are split by how often they are written" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const parsed = try parse(arena.allocator(), reflectionFor("scene"));

    // Three sets, one per rewrite frequency. The spaces are taken from the
    // renderer's own constants rather than repeated as literals, because those
    // constants are what the pipeline layout and the two binds are built from:
    // a literal here would agree with the shader while the Vulkan side moved.
    const camera = find(parsed.parameters, "camera") orelse return error.MissingCameraBlock;
    const instances = find(parsed.parameters, "instances") orelse return error.MissingInstances;
    const materials = find(parsed.parameters, "materials") orelse return error.MissingMaterialBuffer;
    const base_colour = find(parsed.parameters, "base_colour") orelse return error.MissingBaseColourTexture;
    const metallic_roughness = find(parsed.parameters, "metallic_roughness") orelse
        return error.MissingMetallicRoughnessTexture;
    const normal_texture = find(parsed.parameters, "normal_texture") orelse
        return error.MissingNormalTexture;
    const emissive_texture = find(parsed.parameters, "emissive_texture") orelse
        return error.MissingEmissiveTexture;
    const occlusion_texture = find(parsed.parameters, "occlusion_texture") orelse
        return error.MissingOcclusionTexture;

    try testing.expectEqual(@as(u32, gpu.frame_set_index), camera.binding.space);
    try testing.expectEqual(@as(u32, 0), camera.binding.index);
    try testing.expectEqual(@as(u32, gpu.frame_set_index), instances.binding.space);
    try testing.expectEqual(@as(u32, 1), instances.binding.index);
    try testing.expectEqual(@as(u32, gpu.scene_set_index), materials.binding.space);
    try testing.expectEqual(@as(u32, gpu.MaterialArrayBindings[0].slot), materials.binding.index);
    try testing.expectEqual(@as(u32, gpu.material_set_index), base_colour.binding.space);
    try testing.expectEqual(@as(u32, gpu.RendererMaterialBindings[0].slot), base_colour.binding.index);
    try testing.expectEqual(@as(u32, gpu.material_set_index), metallic_roughness.binding.space);
    try testing.expectEqual(
        @as(u32, gpu.RendererMaterialBindings[1].slot),
        metallic_roughness.binding.index,
    );
    try testing.expectEqual(@as(u32, gpu.material_set_index), normal_texture.binding.space);
    try testing.expectEqual(
        @as(u32, gpu.RendererMaterialBindings[2].slot),
        normal_texture.binding.index,
    );
    try testing.expectEqual(@as(u32, gpu.material_set_index), emissive_texture.binding.space);
    try testing.expectEqual(
        @as(u32, gpu.RendererMaterialBindings[3].slot),
        emissive_texture.binding.index,
    );
    try testing.expectEqual(@as(u32, gpu.material_set_index), occlusion_texture.binding.space);
    try testing.expectEqual(
        @as(u32, gpu.RendererMaterialBindings[4].slot),
        occlusion_texture.binding.index,
    );

    // The descriptor type the shader asks for has to be the one the set layout
    // declares. A storage buffer bound as a uniform one is a validation error at
    // draw time and nothing sooner.
    try testing.expectEqual(gpu.MaterialArrayBindings[0].kind, try descriptorTypeOf(materials));
    try testing.expectEqual(gpu.RendererMaterialBindings[0].kind, try descriptorTypeOf(base_colour));
    try testing.expectEqual(
        gpu.RendererMaterialBindings[1].kind,
        try descriptorTypeOf(metallic_roughness),
    );
    try testing.expectEqual(
        gpu.RendererMaterialBindings[2].kind,
        try descriptorTypeOf(normal_texture),
    );
    try testing.expectEqual(
        gpu.RendererMaterialBindings[3].kind,
        try descriptorTypeOf(emissive_texture),
    );
    try testing.expectEqual(
        gpu.RendererMaterialBindings[4].kind,
        try descriptorTypeOf(occlusion_texture),
    );

    var material_binding_count: usize = 0;
    var scene_binding_count: usize = 0;
    for (parsed.parameters) |parameter| {
        if (parameter.binding.space == gpu.material_set_index) material_binding_count += 1;
        if (parameter.binding.space == gpu.scene_set_index) scene_binding_count += 1;
    }
    try testing.expectEqual(gpu.RendererMaterialBindings.len, material_binding_count);
    // The scene set is assembled from two binding lists in two files. A slot
    // added to one of them and not declared in the shader leaves a descriptor
    // the layout reserves and nothing reads; the reverse is a read of a
    // descriptor that was never written.
    try testing.expectEqual(gpu.SceneSetBindings.len, scene_binding_count);
}

// The environment's three descriptors, checked against the same binding list the
// set layout is built from. The shape matters as much as the slot: a cubemap
// declared as a 2D sampler takes the same descriptor type and the same slot, and
// differs only here and at the device.
test "the environment is three descriptors of the shapes the maps have" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const parsed = try parse(arena.allocator(), reflectionFor("scene"));

    const lambertian = find(parsed.parameters, "lambertian_environment") orelse
        return error.MissingLambertianEnvironment;
    const ggx = find(parsed.parameters, "ggx_environment") orelse
        return error.MissingGgxEnvironment;
    const lut = find(parsed.parameters, "ggx_lut") orelse return error.MissingGgxLut;

    // Slots 1, 2 and 3 of the scene set, continuing after the material array.
    // The indices come from the environment's own list rather than as literals.
    const expected = [_]gpu.DescriptorBinding{
        gpu.environment_bindings[0],
        gpu.environment_bindings[1],
        gpu.environment_bindings[2],
    };
    for ([_]Reflection.Parameter{ lambertian, ggx, lut }, expected) |parameter, binding| {
        try testing.expectEqual(@as(u32, gpu.scene_set_index), parameter.binding.space);
        try testing.expectEqual(binding.slot, parameter.binding.index);
        try testing.expectEqual(binding.kind, try descriptorTypeOf(parameter));
    }

    // Both environment maps are cubes and the lookup table is not. Sampling a
    // 2D image through a cube declaration reads a direction as a coordinate
    // pair, which returns plausible values from the wrong place.
    try testing.expectEqualStrings("textureCube", lambertian.type.baseShape);
    try testing.expectEqualStrings("textureCube", ggx.type.baseShape);
    try testing.expectEqualStrings("texture2D", lut.type.baseShape);
}

// The GGX chain's depth is a property of the file that was loaded, and the
// shader turns roughness into a level with it. Carrying it as a constant beside
// the shader is the same class of drift as `min_roughness`, except that a wrong
// value here returns a correctly filtered sample of the wrong roughness, which
// reads as the wrong material. The query is what removes the constant, so its
// presence in the emitted words is the thing worth pinning.
test "the prefiltered level count is queried from the descriptor" {
    const words = gpu.Shaders.scene.spirv;
    var queries: usize = 0;
    var index: usize = 5;
    while (index < words.len) {
        const opcode: u16 = @truncate(words[index]);
        const length: u16 = @intCast(words[index] >> 16);
        if (length == 0) break;
        // SPIR-V specification, 3.49.10: OpImageQueryLevels is opcode 106.
        if (opcode == 106) queries += 1;
        index += length;
    }
    try testing.expectEqual(@as(usize, 1), queries);
}

test "the material record strides at the size the host packs" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const parsed = try parse(arena.allocator(), reflectionFor("scene"));
    const materials = find(parsed.parameters, "materials") orelse return error.MissingMaterialBuffer;

    // A structured buffer is stepped through by its element's own length, so a
    // record shorter than the host's reads every material after the first from
    // the wrong offset and leaves the first one looking correct. `expectMirrors`
    // holds that length, every field offset and the field count at once, which
    // is what keeps the unread `tex` tail from being dropped as dead weight.
    const element = materials.type.resultType orelse return error.MissingElementType;
    try expectMirrors(gpu.MaterialData, element);
}

test "every shader that ships has reflection beside it" {
    // The build emits both from one slangc invocation, so a module without
    // reflection means the two lists went out of step.
    for (gpu.Shaders.all) |module| {
        var found = false;
        for (gpu.Shaders.reflection) |entry| {
            if (std.mem.eql(u8, entry.name, module.name)) found = true;
        }
        try testing.expect(found);
    }
    try testing.expectEqual(gpu.Shaders.all.len, gpu.Shaders.reflection.len);
}

test "every entry point a module declares is in the module's words" {
    // Read out of the emitted SPIR-V and not out of the reflection JSON. The
    // JSON reports the source name whatever the module was given, so it cannot
    // see a rename: a module with one entry point has it emitted as `main`
    // unless slangc is passed -fvk-use-entrypoint-name, and the JSON says
    // `morphMain` either way. That difference reaches nothing until pipeline
    // creation, which fails with the name not found. Measured on 2026-08-08,
    // after it did.
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    for (gpu.Shaders.all) |module| {
        var emitted: std.ArrayList([]const u8) = .empty;
        try entryPointNames(module.spirv, &emitted, arena.allocator());

        var declared_count: usize = 0;
        for ([_]?[*:0]const u8{
            module.vertex_entry,
            module.skinned_vertex_entry,
            module.fragment_entry,
            module.compute_entry,
        }) |candidate| {
            const name = std.mem.span(candidate orelse continue);
            declared_count += 1;

            const found = for (emitted.items) |present| {
                if (std.mem.eql(u8, present, name)) break true;
            } else false;
            if (!found) return error.EntryPointNotInModule;
        }

        // Both directions. An entry point the module carries and shaders.zig
        // does not name is one no pipeline can select, which is a shader
        // written and never reached.
        try testing.expectEqual(declared_count, emitted.items.len);
    }
}

test "every entry point the compiler reports is at the stage the host expects" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const Declared = struct { name: ?[*:0]const u8, stage: []const u8 };
    for (gpu.Shaders.all) |module| {
        const parsed = try parse(arena.allocator(), reflectionFor(module.name));
        for ([_]Declared{
            .{ .name = module.vertex_entry, .stage = "vertex" },
            .{ .name = module.skinned_vertex_entry, .stage = "vertex" },
            .{ .name = module.fragment_entry, .stage = "fragment" },
            .{ .name = module.compute_entry, .stage = "compute" },
        }) |declared| {
            const name = std.mem.span(declared.name orelse continue);
            const found = entryPoint(parsed, name) orelse return error.EntryPointMissing;
            try testing.expectEqualStrings(declared.stage, found.stage);
        }
    }
}

// SPIR-V specification, 3.42.4 Mode-Setting Instructions: OpEntryPoint is
// opcode 15, and its operands are the execution model, the function id, then
// the name as a literal string. A literal string is four characters per word
// and NUL-terminated, so the name ends at the first zero byte.
const op_entry_point: u32 = 15;

fn entryPointNames(
    spirv: []const u32,
    out: *std.ArrayList([]const u8),
    allocator: std.mem.Allocator,
) !void {
    var index: usize = 5;
    while (index < spirv.len) : (index += spirv[index] >> 16) {
        const length = spirv[index] >> 16;
        if ((spirv[index] & 0xFFFF) != op_entry_point) continue;

        const literal = spirv[index + 3 .. index + length];
        const bytes = try allocator.alloc(u8, literal.len * @sizeOf(u32));
        for (literal, 0..) |word, position|
            std.mem.writeInt(u32, bytes[position * 4 ..][0..4], word, .little);
        try out.append(allocator, bytes[0 .. std.mem.indexOfScalar(u8, bytes, 0) orelse bytes.len]);
    }
}

// SPIR-V decoration values, read out of a compiled probe beside its
// disassembly rather than quoted: `OpDecorate` is opcode 71, its first operand
// is the target and its second the decoration, `Location` is 30 and `Flat` is
// 14. Recheck by compiling any shader to both `spirv` and `spirv-asm` and
// matching a `Flat` line against the words at the same position.
const op_decorate: u32 = 71;
const decoration_location: u32 = 30;
const decoration_flat: u32 = 14;

// Every target the module decorates with `Flat`, and the location each of them
// was given. Slang's reflection does not report an interpolation mode at all,
// so the emitted words are the only account of it there is.
fn flatLocations(spirv: []const u32, out: *std.ArrayList(u32), allocator: std.mem.Allocator) !void {
    var flat: std.ArrayList(u32) = .empty;
    defer flat.deinit(allocator);
    var located: std.ArrayList([2]u32) = .empty;
    defer located.deinit(allocator);

    // SPIR-V specification, 2.3 Physical Layout: five header words, then
    // instructions whose first word carries the length in its high half.
    var index: usize = 5;
    while (index < spirv.len) : (index += spirv[index] >> 16) {
        const length = spirv[index] >> 16;
        if ((spirv[index] & 0xFFFF) != op_decorate) continue;
        if (length == 3 and spirv[index + 2] == decoration_flat)
            try flat.append(allocator, spirv[index + 1]);
        if (length == 4 and spirv[index + 2] == decoration_location)
            try located.append(allocator, .{ spirv[index + 1], spirv[index + 3] });
    }

    for (flat.items) |target| {
        for (located.items) |pair| {
            if (pair[0] == target) try out.append(allocator, pair[1]);
        }
    }
}

test "the material index reaches the fragment stage flat" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const parsed = try parse(allocator, reflectionFor("scene"));
    const fragment = for (parsed.entryPoints) |entry| {
        if (std.mem.eql(u8, entry.stage, "fragment")) break entry;
    } else return error.MissingFragmentEntry;
    const varying = fragment.parameters[0].type;
    const material = find(varying.fields, "material_index") orelse return error.MissingMaterialIndex;

    // An interpolated index is a fraction between two records everywhere but the
    // vertices, and it reads a neighbouring material or runs off the array.
    // Every entry point in this module carries the varying, each through its own
    // decorated variable: both vertex variants write it and the fragment stage
    // reads it, so the count is the entry-point count and nothing else here is
    // flat.
    var locations: std.ArrayList(u32) = .empty;
    try flatLocations(gpu.Shaders.scene.spirv, &locations, allocator);
    try testing.expectEqual(parsed.entryPoints.len, locations.items.len);
    for (locations.items) |location|
        try testing.expectEqual(material.binding.index, location);
}

// What descriptor type a reflected parameter has to be bound as. Only the
// shapes this engine's shaders use are here; an unrecognised one is a shader
// that grew a binding nothing on this side knows how to declare.
fn descriptorTypeOf(parameter: Reflection.Parameter) !vk.DescriptorType {
    const kind = parameter.type.kind;
    if (std.mem.eql(u8, kind, "constantBuffer")) return .uniform_buffer;
    if (std.mem.eql(u8, kind, "resource")) {
        const shape = parameter.type.baseShape;
        if (std.mem.eql(u8, shape, "structuredBuffer")) return .storage_buffer;
        // Both are the same descriptor type. What differs is the coordinate the
        // shader addresses them with, which the shape check at the call site is
        // what holds.
        if (std.mem.eql(u8, shape, "texture2D")) return .combined_image_sampler;
        if (std.mem.eql(u8, shape, "textureCube")) return .combined_image_sampler;
    }
    return error.UnknownBindingShape;
}

// A dynamic descriptor is the same resource reached through an offset supplied
// at bind time, so it has to match its static counterpart.
fn withoutDynamic(kind: vk.DescriptorType) vk.DescriptorType {
    return switch (kind) {
        .uniform_buffer_dynamic => .uniform_buffer,
        .storage_buffer_dynamic => .storage_buffer,
        else => kind,
    };
}

test "the frame set declares exactly what set zero of the shader holds" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const parsed = try parse(arena.allocator(), reflectionFor("scene"));

    // Every binding in set zero has an entry in the table, with the same slot
    // and a descriptor type that matches what the shader declared there. This
    // is the pair a picture cannot tell apart from a wrong matrix: a set layout
    // that disagrees with the shader draws nothing and reports nothing.
    var counted: usize = 0;
    for (parsed.parameters) |parameter| {
        if (parameter.binding.space != 0) continue;
        counted += 1;

        const declared = for (gpu.FrameSetBindings) |binding| {
            if (binding.slot == parameter.binding.index) break binding;
        } else return error.BindingNotDeclared;

        try testing.expectEqual(
            try descriptorTypeOf(parameter),
            withoutDynamic(declared.kind),
        );
    }
    try testing.expectEqual(gpu.FrameSetBindings.len, counted);
}

test "both frame bindings carry a dynamic offset" {
    // The whole reason one set serves every frame. A static descriptor here
    // would bind frame zero for every frame and never fail a validation layer.
    for (gpu.FrameSetBindings) |binding| {
        try testing.expect(binding.kind != withoutDynamic(binding.kind));
    }
}

test "the post set declares exactly the fullscreen descriptors" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const parsed = try parse(arena.allocator(), reflectionFor("fullscreen"));
    var descriptor_count: usize = 0;
    for (parsed.parameters) |parameter| {
        if (std.mem.eql(u8, parameter.binding.kind, "pushConstantBuffer")) continue;
        descriptor_count += 1;

        const declared = for (gpu.PostBindings) |binding| {
            if (binding.slot == parameter.binding.index) break binding;
        } else return error.BindingNotDeclared;
        try testing.expectEqual(try descriptorTypeOf(parameter), declared.kind);
    }
    try testing.expectEqual(gpu.PostBindings.len, descriptor_count);
}

test "an instance is what the shader steps through, field by field" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const parsed = try parse(arena.allocator(), reflectionFor("scene"));
    const instances = find(parsed.parameters, "instances") orelse return error.MissingInstances;

    // The stride of a structured buffer is its element's size, and a mirror of a
    // different width reads the first instance correctly and every one after it
    // from the wrong offset. The field list is checked as well as the width,
    // because the matrix and the joint base could be the right eighty bytes in
    // the wrong order and the first instance would still look right.
    const element = instances.type.resultType orelse return error.NotAStructuredBuffer;
    try expectMirrors(gpu.Instance, element);
}

test "a joint matrix is what the shader steps through" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const parsed = try parse(arena.allocator(), reflectionFor("scene"));
    const joints = find(parsed.parameters, "joints") orelse return error.MissingJoints;

    // Set zero, slot three, and a stride that matches what a pose is copied in
    // as. The stride is the one thing a wrong joint base cannot be told apart
    // from on screen: both come out as a mesh that moves in the wrong way.
    try testing.expectEqual(@as(u32, 0), joints.binding.space);
    try testing.expectEqual(@as(u32, 3), joints.binding.index);

    const element = joints.type.resultType orelse return error.NotAStructuredBuffer;
    try testing.expectEqual(
        @as(u32, @intCast(@sizeOf(gpu.Joint))),
        try uniformSize(element),
    );
}

fn entryPoint(parsed: Reflection, name: []const u8) ?Reflection.EntryPoint {
    for (parsed.entryPoints) |candidate| {
        if (std.mem.eql(u8, candidate.name, name)) return candidate;
    }
    return null;
}

// The location a semantic landed on, which is what a vertex input attribute has
// to name to reach it.
fn varyingLocationIn(parameters: []const Reflection.Parameter, semantic: []const u8) !u32 {
    for (parameters) |parameter| {
        if (!std.mem.eql(u8, parameter.semanticName, semantic)) continue;
        if (!std.mem.eql(u8, parameter.binding.kind, "varyingInput")) return error.NotAVaryingInput;
        return parameter.binding.index;
    }
    return error.MissingSemantic;
}

fn varyingLocation(entry: Reflection.EntryPoint, semantic: []const u8) !u32 {
    return varyingLocationIn(entry.parameters, semantic);
}

// Where the mesh's own layout puts an attribute, found by the offset of the
// field it carries rather than by its position in the list.
fn baseLocation(comptime field: []const u8) !u32 {
    for (gpu.GpuVertex.attribute_descriptions) |attribute| {
        if (attribute.offset == @offsetOf(gpu.GpuVertex, field)) return attribute.location;
    }
    return error.NoAttributeForField;
}

fn skinLocation(comptime field: []const u8) !u32 {
    for (gpu.GpuSkinVertex.attribute_descriptions) |attribute| {
        if (attribute.offset == @offsetOf(gpu.GpuSkinVertex, field)) return attribute.location;
    }
    return error.NoAttributeForField;
}

test "the skinned entry point takes the skinning stream at the locations the mesh declares" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const parsed = try parse(arena.allocator(), reflectionFor("scene"));

    // Resolved through the name the renderer creates the pipeline with, not a
    // literal. Both entry points are vertex stage and both exist, so nothing
    // else here can tell the two apart: swapping the module's two names would
    // build the skinned pipeline over the unskinned shader and report nothing.
    const skinned = entryPoint(
        parsed,
        std.mem.span(gpu.Shaders.scene.skinned_vertex_entry.?),
    ) orelse return error.MissingSkinnedEntryPoint;
    try testing.expectEqualStrings("vertex", skinned.stage);

    // Locations are assigned in declaration order and nothing states them in
    // the shader source, so this is the pair that drifts silently: the vertex
    // input feeds an attribute to a location the shader gave to something else,
    // and the layer sees two sides that each agree with themselves.
    try testing.expectEqual(try skinLocation("joints"), try varyingLocation(skinned, "JOINTS"));
    try testing.expectEqual(try skinLocation("weights"), try varyingLocation(skinned, "WEIGHTS"));

    // The base stream is unmoved by the two that follow it. If the skin
    // attributes had been declared first, every location in `GpuVertex` would
    // be off by two and the mesh would draw with its normals in its positions.
    const base = find(skinned.parameters, "vertex") orelse return error.MissingBaseVertex;
    try testing.expectEqual(@as(u32, 0), base.binding.index);
    try testing.expectEqual(
        @as(u32, gpu.GpuVertex.attribute_descriptions.len),
        base.binding.count,
    );
    try testing.expectEqual(
        try baseLocation("tangent"),
        try varyingLocationIn(base.type.fields, "TANGENT"),
    );
}

test "the prepass dispatches over the workgroup the shader declares" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const parsed = try parse(arena.allocator(), reflectionFor("morph"));
    const entry = entryPoint(
        parsed,
        std.mem.span(gpu.Shaders.morph.compute_entry.?),
    ) orelse return error.MissingComputeEntryPoint;

    // Vulkan takes the workgroup size from the module, and the host divides the
    // vertex count by its own constant to size the dispatch. Nothing relates the
    // two: a shader declaring 32 against a host dividing by 64 dispatches half
    // the groups a mesh needs and leaves its second half at the bind shape, with
    // no error anywhere. This is the only thing that holds them together.
    try testing.expectEqualSlices(u32, &.{ gpu.morphGroupSize, 1, 1 }, entry.threadGroupSize);
}

test "the prepass set declares exactly the descriptors the shader reads" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const parsed = try parse(arena.allocator(), reflectionFor("morph"));

    var counted: usize = 0;
    for (parsed.parameters) |parameter| {
        if (std.mem.eql(u8, parameter.binding.kind, "pushConstantBuffer")) continue;
        counted += 1;
        // One set, so a binding that landed in another space is a shader whose
        // layout the pipeline was not built from.
        try testing.expectEqual(@as(u32, 0), parameter.binding.space);

        const declared = for (gpu.morph_bindings) |binding| {
            if (binding.slot == parameter.binding.index) break binding;
        } else return error.BindingNotDeclared;
        try testing.expectEqual(
            try descriptorTypeOf(parameter),
            withoutDynamic(declared.kind),
        );
    }
    try testing.expectEqual(gpu.morph_bindings.len, counted);
}

test "the prepass reads and writes vertices of the layout the mesh packs" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const parsed = try parse(arena.allocator(), reflectionFor("morph"));

    // Both ends are the same GpuVertex, which is the whole reason the substitute
    // buffer needs no second vertex input. A float3 in the shader's mirror
    // aligns to sixteen bytes and strides the element by thirty-two against the
    // host's twenty-four, so the second vertex on would be read and written from
    // the wrong offset; measured against this reflection, not derived.
    for ([_][]const u8{ "source", "morphed" }) |name| {
        const parameter = find(parsed.parameters, name) orelse return error.MissingVertexBuffer;
        const element = parameter.type.resultType orelse return error.NotAStructuredBuffer;
        try testing.expectEqual(
            @as(u32, @sizeOf(gpu.GpuVertex)),
            try uniformSize(element),
        );
    }

    const deltas = find(parsed.parameters, "deltas") orelse return error.MissingDeltas;
    const element = deltas.type.resultType orelse return error.NotAStructuredBuffer;
    // Six floats, which is what mesh/resource.zig interleaves a position and a
    // normal displacement into.
    try testing.expectEqual(@as(u32, 24), try uniformSize(element));
}

test "the prepass push block is what the shader reads, field by field" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const parsed = try parse(arena.allocator(), reflectionFor("morph"));
    const push = find(parsed.parameters, "push") orelse return error.MissingPushConstants;
    try testing.expectEqualStrings("pushConstantBuffer", push.binding.kind);
    try expectMirrors(gpu.MorphPushConstants, try block(push));

    // The range the host pushes has to cover the block the shader reads. A short
    // range leaves the tail as whatever the previous pipeline pushed.
    try testing.expectEqual(
        try uniformSize(try block(push)),
        gpu.morphPushConstantRange.size,
    );
}

test "both vertex entry points are in the module the pipeline names" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();

    const parsed = try parse(arena.allocator(), reflectionFor("scene"));

    // One file, one module, three entry points. The skinned and unskinned
    // variants share every binding in set zero, which is what keeps them a
    // pipeline difference rather than a second descriptor layout.
    try testing.expect(entryPoint(parsed, "vertexMain") != null);
    try testing.expect(entryPoint(parsed, "skinnedVertexMain") != null);
    try testing.expect(entryPoint(parsed, "fragmentMain") != null);
}
