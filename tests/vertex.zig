const std = @import("std");
const gpu = @import("lenore-gpu");
const res = @import("lenore-resources");

const testing = std.testing;
const Vertex3D = res.Vertex3D;
const GpuVertex = gpu.GpuVertex;

// Snorm quantisation of a full-scale component. Vulkan converts a signed
// normalised value as round(clamp(v, -1, 1) * (2^(b-1) - 1)), so ten bits reach
// 511 and the negative end is that value in two's complement over ten bits.
const snorm10_one: u32 = 511;
const snorm10_minus_one: u32 = 0x3FF - 511 + 1;
const snorm2_one: u32 = 1;
const snorm2_minus_one: u32 = 3;

fn vertex(fields: anytype) Vertex3D {
    var out: Vertex3D = .{
        .position = .{ 0, 0, 0 },
        .normal = .{ 0, 1, 0 },
        .uv = .{ 0, 0 },
        .tangent = .{ 1, 0, 0, 1 },
    };
    inline for (@typeInfo(@TypeOf(fields)).@"struct".fields) |field| {
        @field(out, field.name) = @field(fields, field.name);
    }
    return out;
}

// Resolving the GPU types is what makes their comptime size asserts run.
test "the GPU layouts are what a pipeline binds" {
    try testing.expectEqual(@as(usize, 24), @sizeOf(GpuVertex));
    try testing.expectEqual(@as(usize, 16), @sizeOf(gpu.GpuSkinVertex));
    try testing.expectEqual(@as(usize, 4), @sizeOf(gpu.GpuColourVertex));
    try testing.expectEqual(@as(usize, 4), @sizeOf(gpu.GpuUv1Vertex));

    try testing.expectEqual(@as(usize, 0), @offsetOf(GpuVertex, "position"));
    try testing.expectEqual(@as(usize, 12), @offsetOf(GpuVertex, "uv"));
    try testing.expectEqual(@as(usize, 16), @offsetOf(GpuVertex, "normal"));
    try testing.expectEqual(@as(usize, 20), @offsetOf(GpuVertex, "tangent"));

    // Locations are consecutive across bindings, because a shader declares one
    // input list regardless of which streams a mesh supplies.
    try testing.expectEqual(@as(u32, 0), GpuVertex.attribute_descriptions[0].location);
    try testing.expectEqual(@as(u32, 3), GpuVertex.attribute_descriptions[3].location);
    try testing.expectEqual(@as(u32, 4), gpu.GpuSkinVertex.attribute_descriptions[0].location);
    try testing.expectEqual(@as(u32, 6), gpu.GpuColourVertex.attribute_descriptions[0].location);
    try testing.expectEqual(@as(u32, 7), gpu.GpuUv1Vertex.attribute_descriptions[0].location);

    try testing.expectEqual(@as(u32, 24), GpuVertex.binding_description.stride);
    try testing.expectEqual(@as(u32, 1), gpu.GpuSkinVertex.binding_description.binding);
}

test "a direction packs into the ten-bit lanes at full scale" {
    try testing.expectEqual(
        snorm10_one,
        gpu.packSnorm3x10_1x2(.{ 1, 0, 0, 0 }),
    );
    try testing.expectEqual(
        snorm10_one << 10,
        gpu.packSnorm3x10_1x2(.{ 0, 1, 0, 0 }),
    );
    try testing.expectEqual(
        snorm10_one << 20,
        gpu.packSnorm3x10_1x2(.{ 0, 0, 1, 0 }),
    );
    try testing.expectEqual(
        snorm10_minus_one,
        gpu.packSnorm3x10_1x2(.{ -1, 0, 0, 0 }),
    );
    // Out of range clamps rather than wrapping into another lane.
    try testing.expectEqual(
        snorm10_one,
        gpu.packSnorm3x10_1x2(.{ 4.0, 0, 0, 0 }),
    );
}

test "handedness survives as its sign in the two-bit lane" {
    const right = gpu.packVertex(&vertex(.{ .tangent = .{ 1, 0, 0, 1 } }));
    const left = gpu.packVertex(&vertex(.{ .tangent = .{ 1, 0, 0, -1 } }));

    try testing.expectEqual(snorm2_one, right.tangent >> 30);
    try testing.expectEqual(snorm2_minus_one, left.tangent >> 30);

    // A tangent whose handedness is not a number takes the positive branch
    // rather than producing an undefined conversion.
    const unknown = gpu.packVertex(&vertex(.{
        .tangent = .{ 1, 0, 0, std.math.nan(f32) },
    }));
    try testing.expectEqual(snorm2_one, unknown.tangent >> 30);
}

test "a scaled direction is normalised before packing" {
    // A normal matrix can leave a direction longer than one. Packing it
    // unnormalised would clamp every component to full scale and flatten it.
    const scaled = gpu.packVertex(&vertex(.{ .normal = .{ 0, 7, 0 } }));
    try testing.expectEqual(snorm10_one << 10, scaled.normal);

    const unit = gpu.packVertex(&vertex(.{ .normal = .{ 0, 1, 0 } }));
    try testing.expectEqual(unit.normal, scaled.normal);
}

// The packing runs on asset data nobody validated, and in the shipping build
// there is no safety check between it and undefined behaviour. Both degenerate
// cases have to land on the fallback by construction.
test "degenerate and non-numeric directions fall back" {
    const expected = gpu.packVertex(&vertex(.{ .normal = .{ 0, 1, 0 } })).normal;

    const zero = gpu.packVertex(&vertex(.{ .normal = .{ 0, 0, 0 } }));
    try testing.expectEqual(expected, zero.normal);

    const nan = std.math.nan(f32);
    const not_a_number = gpu.packVertex(&vertex(.{ .normal = .{ nan, nan, nan } }));
    try testing.expectEqual(expected, not_a_number.normal);

    // An infinite component divides to NaN, which the clamp downstream would
    // turn into full scale and hand the shader a direction that is not unit
    // length.
    const infinite = gpu.packVertex(&vertex(.{
        .normal = .{ std.math.inf(f32), 0, 0 },
    }));
    try testing.expectEqual(expected, infinite.normal);

    const both_infinite = gpu.packVertex(&vertex(.{
        .normal = .{ std.math.inf(f32), std.math.inf(f32), 0 },
    }));
    try testing.expectEqual(expected, both_infinite.normal);
}

test "positions and UVs keep their precision classes" {
    const packed_vertex = gpu.packVertex(&vertex(.{
        .position = .{ 1.5, -2.25, 3.125 },
        .uv = .{ 0.25, 0.75 },
    }));

    // Position is f32 through, so these are exact.
    try testing.expectEqual([3]f32{ 1.5, -2.25, 3.125 }, packed_vertex.position);
    // These UVs are representable in f16, so they are exact too.
    try testing.expectEqual(@as(f16, 0.25), packed_vertex.uv[0]);
    try testing.expectEqual(@as(f16, 0.75), packed_vertex.uv[1]);
}

test "joint weights quantise to unorm16 and tolerate bad input" {
    const nan = std.math.nan(f32);
    const skin = gpu.packSkinVertex(&vertex(.{
        .joints = .{ 0, 7, 65535, 3 },
        .weights = .{ 0.0, 1.0, 0.5, nan },
    }));

    try testing.expectEqual([4]u16{ 0, 7, 65535, 3 }, skin.joints);
    try testing.expectEqual(@as(u16, 0), skin.weights[0]);
    try testing.expectEqual(@as(u16, 65535), skin.weights[1]);
    // 0.5 * 65535 is 32767.5, and rounding away from zero gives 32768.
    try testing.expectEqual(@as(u16, 32768), skin.weights[2]);
    // Clamping resolves a non-number to the upper bound rather than leaving the
    // conversion undefined.
    try testing.expectEqual(@as(u16, 65535), skin.weights[3]);
}

test "the optional streams are straight copies" {
    const colour = gpu.packColourVertex(&vertex(.{ .colour = .{ 1, 2, 3, 4 } }));
    try testing.expectEqual([4]u8{ 1, 2, 3, 4 }, colour.colour);

    const uv1 = gpu.packUv1Vertex(&vertex(.{ .uv1 = .{ 0.5, 0.125 } }));
    try testing.expectEqual(@as(f16, 0.5), uv1.uv1[0]);
    try testing.expectEqual(@as(f16, 0.125), uv1.uv1[1]);
}
