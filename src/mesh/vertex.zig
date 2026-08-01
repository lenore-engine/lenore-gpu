const std = @import("std");
const vk = @import("vulkan");

pub const Vec2 = @Vector(2, f32);
pub const Vec3 = @Vector(3, f32);
pub const Vec4 = @Vector(4, f32);

// The vertex as an asset produces it and as load-time maths operates on it:
// full precision, so merging, transform baking and bounds lose nothing. It is
// packed into the GPU layouts once, at upload, and never uploaded as it stands.
pub const Vertex3D = struct {
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    // xyz is the tangent, w the handedness that reconstructs the bitangent.
    tangent: Vec4,

    // COLOR_0, already normalised to unorm8 rgba. glTF 2.0 specification, 3.7.2,
    // mesh attribute table: the attribute is VEC3 or VEC4 in float, normalised
    // unsigned byte or normalised unsigned short, and every component is clamped
    // to zero through one, so all of them fold to this at parse time. White is
    // the identity tint, so a mesh without the attribute carries a stream that
    // changes nothing.
    colour: [4]u8 = .{ 255, 255, 255, 255 },

    // TEXCOORD_1. A material selects UV set 0 or 1 per texture slot, so a mesh
    // without this attribute is only wrong if an asset selects set 1 for it.
    uv1: Vec2 = .{ 0.0, 0.0 },

    // glTF 2.0 specification, 3.7.2: JOINTS_n and WEIGHTS_n are VEC4, so one
    // attribute set holds four joints per vertex.
    joints: @Vector(4, u16) = @splat(0),
    weights: Vec4 = @splat(0.0),
};

// The optional streams a mesh carries beyond the mandatory binding 0. Each flag
// is one optional binding, described once by the corresponding GPU struct below.
// Adding a stream is a field here and a struct there, rather than a branch in
// every place that touches vertices.
pub const VertexStreams = packed struct(u3) {
    skinned: bool = false,
    colour: bool = false,
    uv1: bool = false,

    pub fn index(self: VertexStreams) u3 {
        return @bitCast(self);
    }
};

// Binding 0, present on every mesh. Positions stay f32 because geometry
// precision is visible; UVs are f16, and normal and tangent are 10-10-10-2
// snorm.
//
// Ten signed bits give 511 steps across zero to one, so a component resolves to
// about 1/511, and near the equator of the unit sphere that is an angle of
// roughly a ninth of a degree.
pub const GpuVertex = extern struct {
    position: [3]f32,
    uv: [2]f16,
    // 10-10-10-2 snorm, w unused.
    normal: u32,
    // 10-10-10-2 snorm, w carries the handedness as plus or minus one.
    tangent: u32,

    pub const binding_description: vk.VertexInputBindingDescription = .{
        .binding = 0,
        .stride = @sizeOf(GpuVertex),
        .input_rate = .vertex,
    };

    pub const attribute_descriptions = [_]vk.VertexInputAttributeDescription{
        .{
            .binding = 0,
            .location = 0,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(GpuVertex, "position"),
        },
        .{
            .binding = 0,
            .location = 1,
            .format = .r16g16_sfloat,
            .offset = @offsetOf(GpuVertex, "uv"),
        },
        .{
            .binding = 0,
            .location = 2,
            .format = .a2b10g10r10_snorm_pack32,
            .offset = @offsetOf(GpuVertex, "normal"),
        },
        .{
            .binding = 0,
            .location = 3,
            .format = .a2b10g10r10_snorm_pack32,
            .offset = @offsetOf(GpuVertex, "tangent"),
        },
    };

    comptime {
        std.debug.assert(@sizeOf(GpuVertex) == 24);
    }
};

// Binding 1, fetched by skinned draws only, so a static mesh carries no
// skinning bytes at all. Weights are unorm16 rather than f16 because they live
// in zero to one and sum to one: unorm spends all 65536 codes evenly across that
// range, while f16 spends half of its on values below 1/64 and leaves about
// 1/2048 near one.
pub const GpuSkinVertex = extern struct {
    joints: [4]u16,
    weights: [4]u16,

    pub const binding_description: vk.VertexInputBindingDescription = .{
        .binding = 1,
        .stride = @sizeOf(GpuSkinVertex),
        .input_rate = .vertex,
    };

    pub const attribute_descriptions = [_]vk.VertexInputAttributeDescription{
        .{
            .binding = 1,
            .location = 4,
            .format = .r16g16b16a16_uint,
            .offset = @offsetOf(GpuSkinVertex, "joints"),
        },
        .{
            .binding = 1,
            .location = 5,
            .format = .r16g16b16a16_unorm,
            .offset = @offsetOf(GpuSkinVertex, "weights"),
        },
    };

    comptime {
        std.debug.assert(@sizeOf(GpuSkinVertex) == 16);
    }
};

// Binding 2, fetched only by meshes that declared COLOR_0. unorm8 carries what
// the attribute means: glTF 2.0 specification, 3.7.2, clamps every component of
// COLOR_0 to zero through one, so it is a display-range multiplier.
pub const GpuColourVertex = extern struct {
    colour: [4]u8,

    pub const binding_description: vk.VertexInputBindingDescription = .{
        .binding = 2,
        .stride = @sizeOf(GpuColourVertex),
        .input_rate = .vertex,
    };

    pub const attribute_descriptions = [_]vk.VertexInputAttributeDescription{
        .{
            .binding = 2,
            .location = 6,
            .format = .r8g8b8a8_unorm,
            .offset = @offsetOf(GpuColourVertex, "colour"),
        },
    };

    comptime {
        std.debug.assert(@sizeOf(GpuColourVertex) == 4);
    }
};

// Binding 3, fetched only by meshes that declared TEXCOORD_1. Same precision as
// the binding 0 UV, and the same large-coordinate caveat.
pub const GpuUv1Vertex = extern struct {
    uv1: [2]f16,

    pub const binding_description: vk.VertexInputBindingDescription = .{
        .binding = 3,
        .stride = @sizeOf(GpuUv1Vertex),
        .input_rate = .vertex,
    };

    pub const attribute_descriptions = [_]vk.VertexInputAttributeDescription{
        .{
            .binding = 3,
            .location = 7,
            .format = .r16g16_sfloat,
            .offset = @offsetOf(GpuUv1Vertex, "uv1"),
        },
    };

    comptime {
        std.debug.assert(@sizeOf(GpuUv1Vertex) == 4);
    }
};

// Above this magnitude an f16 UV is worth inspecting. It is not a range limit:
// the format's largest finite value is far above it. f16 carries a 10-bit
// mantissa, so in the binade starting at 16 consecutive values differ by
// 16 * 2^-10, which is 1/64 of a UV unit, or 16 texels on a repeating
// 1024-texel texture, and the gap doubles with every binade above. Large tiled
// coordinates are valid glTF, so this drives a diagnostic rather than a
// rejection.
pub const uv_f16_inspection_threshold: f32 = 16.0;

// A direction of length zero cannot be normalised, and this is the length below
// which one is treated as degenerate rather than divided by.
const degenerate_length: f32 = 1e-8;

pub fn packVertex(vertex: *const Vertex3D) GpuVertex {
    return .{
        .position = .{ vertex.position[0], vertex.position[1], vertex.position[2] },
        .uv = .{ @floatCast(vertex.uv[0]), @floatCast(vertex.uv[1]) },
        .normal = packDirection(vertex.normal, .{ 0, 1, 0 }),
        .tangent = tangent: {
            const direction = normalise(
                .{ vertex.tangent[0], vertex.tangent[1], vertex.tangent[2] },
                .{ 1, 0, 0 },
            );
            // Handedness is stored as plus or minus one in a two-bit snorm, so
            // only its sign survives, which is all it carries. A NaN compares
            // false and lands on the positive branch.
            const handedness: f32 = if (vertex.tangent[3] < 0) -1 else 1;
            break :tangent packSnorm3x10_1x2(.{
                direction[0],
                direction[1],
                direction[2],
                handedness,
            });
        },
    };
}

pub fn packSkinVertex(vertex: *const Vertex3D) GpuSkinVertex {
    return .{
        .joints = vertex.joints,
        .weights = .{
            packUnorm16(vertex.weights[0]),
            packUnorm16(vertex.weights[1]),
            packUnorm16(vertex.weights[2]),
            packUnorm16(vertex.weights[3]),
        },
    };
}

// The interchange colour is already unorm8, normalised at parse time, so this
// is a copy.
pub fn packColourVertex(vertex: *const Vertex3D) GpuColourVertex {
    return .{ .colour = vertex.colour };
}

pub fn packUv1Vertex(vertex: *const Vertex3D) GpuUv1Vertex {
    return .{ .uv1 = .{ @floatCast(vertex.uv1[0]), @floatCast(vertex.uv1[1]) } };
}

// Normalises before packing, so a direction that a normal matrix scaled beyond
// unit length is not clamped flat. A degenerate one becomes the fallback.
pub fn packDirection(direction: Vec3, fallback: Vec3) u32 {
    const unit = normalise(direction, fallback);
    return packSnorm3x10_1x2(.{ unit[0], unit[1], unit[2], 0 });
}

// Vulkan specification, VK_FORMAT_A2B10G10R10_SNORM_PACK32: a single 32-bit
// word with x in bits 0 to 9, y in 10 to 19, z in 20 to 29 and w in 30 to 31.
pub fn packSnorm3x10_1x2(value: [4]f32) u32 {
    return packSnorm(10, value[0]) |
        (packSnorm(10, value[1]) << 10) |
        (packSnorm(10, value[2]) << 20) |
        (packSnorm(2, value[3]) << 30);
}

fn normalise(direction: Vec3, fallback: Vec3) Vec3 {
    const length = @sqrt(@reduce(.Add, direction * direction));
    // Both non-finite cases take the fallback, and neither is redundant. A NaN
    // component makes the length NaN. An infinite one makes it infinite, and
    // dividing by that yields NaN per component, which the clamp downstream
    // turns into full scale: an asset with an infinite normal would otherwise
    // reach the shader as a direction that is not unit length, shading wrongly
    // and reporting nothing.
    if (!std.math.isFinite(length) or length <= degenerate_length) return fallback;
    return direction / @as(Vec3, @splat(length));
}

// Vulkan specification, Fixed-Point Data Conversion: a signed normalised value
// converts as round(clamp(v, -1, 1) * (2^(b-1) - 1)), and the result is stored
// in two's complement.
//
// The clamp is also what makes the conversion below defined for any input.
// Measured on 0.16: std.math.clamp is @max(lower, @min(val, upper)), and Zig's
// @min and @max return the operand that is not NaN, so clamp(NaN, -1, 1) is 1
// rather than NaN. Without that, @intFromFloat would be undefined behaviour in
// the shipping build on asset data nobody validated.
fn packSnorm(comptime bits: u5, value: f32) u32 {
    const scale: f32 = @floatFromInt((@as(i32, 1) << (bits - 1)) - 1);
    const quantised: i32 = @intFromFloat(@round(std.math.clamp(value, -1.0, 1.0) * scale));
    return @as(u32, @bitCast(quantised)) & ((@as(u32, 1) << bits) - 1);
}

// Vulkan specification, Fixed-Point Data Conversion: an unsigned normalised
// value converts as round(clamp(v, 0, 1) * (2^b - 1)). The same NaN reasoning as
// packSnorm applies.
fn packUnorm16(value: f32) u16 {
    return @intFromFloat(@round(std.math.clamp(value, 0.0, 1.0) * 65535.0));
}
