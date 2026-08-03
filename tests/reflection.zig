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

    // Set zero is the frame's and set one is the material's. Moving a binding
    // between them changes how often a descriptor set has to be rewritten,
    // which is the whole reason there are two.
    const camera = find(parsed.parameters, "camera") orelse return error.MissingCameraBlock;
    const instances = find(parsed.parameters, "instances") orelse return error.MissingInstances;
    const base_colour = find(parsed.parameters, "base_colour") orelse return error.MissingTexture;

    try testing.expectEqual(@as(u32, 0), camera.binding.space);
    try testing.expectEqual(@as(u32, 0), camera.binding.index);
    try testing.expectEqual(@as(u32, 0), instances.binding.space);
    try testing.expectEqual(@as(u32, 1), instances.binding.index);
    try testing.expectEqual(@as(u32, 1), base_colour.binding.space);
    try testing.expectEqual(@as(u32, 0), base_colour.binding.index);
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

// What descriptor type a reflected parameter has to be bound as. Only the
// shapes this engine's shaders use are here; an unrecognised one is a shader
// that grew a binding nothing on this side knows how to declare.
fn descriptorTypeOf(parameter: Reflection.Parameter) !vk.DescriptorType {
    const kind = parameter.type.kind;
    if (std.mem.eql(u8, kind, "constantBuffer")) return .uniform_buffer;
    if (std.mem.eql(u8, kind, "resource")) {
        const shape = parameter.type.baseShape;
        if (std.mem.eql(u8, shape, "structuredBuffer")) return .storage_buffer;
        if (std.mem.eql(u8, shape, "texture2D")) return .combined_image_sampler;
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
fn varyingLocation(entry: Reflection.EntryPoint, semantic: []const u8) !u32 {
    for (entry.parameters) |parameter| {
        if (!std.mem.eql(u8, parameter.semanticName, semantic)) continue;
        if (!std.mem.eql(u8, parameter.binding.kind, "varyingInput")) return error.NotAVaryingInput;
        return parameter.binding.index;
    }
    return error.MissingSemantic;
}

// Where the mesh's own layout puts an attribute, found by the offset of the
// field it carries rather than by its position in the list.
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

    // Selected by name at pipeline creation, so a rename here is a pipeline
    // that fails to create rather than anything a picture shows.
    const skinned = entryPoint(parsed, "skinnedVertexMain") orelse return error.MissingSkinnedEntryPoint;
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
