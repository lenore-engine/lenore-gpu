const std = @import("std");
const gpu = @import("lenore-gpu");

const testing = std.testing;
const Vertex3D = gpu.Vertex3D;

fn at(x: f32, y: f32, z: f32) Vertex3D {
    return .{
        .position = .{ x, y, z },
        .normal = .{ 0, 1, 0 },
        .uv = .{ 0, 0 },
        .tangent = .{ 1, 0, 0, 1 },
    };
}

fn expectVec(expected: [3]f32, actual: gpu.Vec3) !void {
    try testing.expectEqual(expected, [3]f32{ actual[0], actual[1], actual[2] });
}

test "the box spans every vertex" {
    const vertices = [_]Vertex3D{
        at(-1, 2, 0),
        at(3, -4, 5),
        at(0, 0, -6),
    };
    const bounds = gpu.Bounds.compute(&vertices);

    try expectVec(.{ -1, -4, -6 }, bounds.box.min);
    try expectVec(.{ 3, 2, 5 }, bounds.box.max);
    try expectVec(.{ 1, -1, -0.5 }, bounds.box.centre());
    try expectVec(.{ 4, 6, 11 }, bounds.box.extent());
}

test "the sphere sits on the box centre and reaches the furthest vertex" {
    // A cube of side two centred on the origin: the corner distance is the
    // square root of three.
    const vertices = [_]Vertex3D{
        at(-1, -1, -1),
        at(1, 1, 1),
        at(-1, 1, -1),
        at(0, 0, 0),
    };
    const bounds = gpu.Bounds.compute(&vertices);

    try expectVec(.{ 0, 0, 0 }, bounds.sphere.centre);
    try testing.expectApproxEqAbs(@sqrt(3.0), bounds.sphere.radius, 1e-6);

    // Every vertex is inside, which is the only property culling depends on.
    for (vertices) |item| {
        const offset = item.position - bounds.sphere.centre;
        const distance = @sqrt(@reduce(.Add, offset * offset));
        try testing.expect(distance <= bounds.sphere.radius + 1e-6);
    }
}

test "the sphere is centred on the box, not on the vertex spread" {
    // Vertices bunched at one end: the centre is the box midpoint regardless,
    // and the radius has to reach the far one from there.
    const vertices = [_]Vertex3D{
        at(0, 0, 0),
        at(0.1, 0, 0),
        at(0.2, 0, 0),
        at(10, 0, 0),
    };
    const bounds = gpu.Bounds.compute(&vertices);

    try expectVec(.{ 5, 0, 0 }, bounds.sphere.centre);
    try testing.expectApproxEqAbs(@as(f32, 5.0), bounds.sphere.radius, 1e-6);
}

test "a single vertex gives a point" {
    const vertices = [_]Vertex3D{at(4, 5, 6)};
    const bounds = gpu.Bounds.compute(&vertices);

    try expectVec(.{ 4, 5, 6 }, bounds.box.min);
    try expectVec(.{ 4, 5, 6 }, bounds.box.max);
    try expectVec(.{ 4, 5, 6 }, bounds.sphere.centre);
    try testing.expectEqual(@as(f32, 0), bounds.sphere.radius);
}

test "an empty mesh is a degenerate box at the origin" {
    const bounds = gpu.Bounds.compute(&.{});

    try expectVec(.{ 0, 0, 0 }, bounds.box.min);
    try expectVec(.{ 0, 0, 0 }, bounds.box.max);
    try expectVec(.{ 0, 0, 0 }, bounds.sphere.centre);
    try testing.expectEqual(@as(f32, 0), bounds.sphere.radius);
}

// Positions come from an asset. A vertex that is not a number must not turn the
// whole mesh's bounds into nothing, which is what would happen if the
// comparisons propagated it.
test "a vertex that is not a number drops out of the bounds" {
    const nan = std.math.nan(f32);
    const vertices = [_]Vertex3D{
        at(nan, nan, nan),
        at(-2, -2, -2),
        at(2, 2, 2),
    };
    const bounds = gpu.Bounds.compute(&vertices);

    try expectVec(.{ -2, -2, -2 }, bounds.box.min);
    try expectVec(.{ 2, 2, 2 }, bounds.box.max);
    try expectVec(.{ 0, 0, 0 }, bounds.sphere.centre);
    try testing.expectApproxEqAbs(@sqrt(12.0), bounds.sphere.radius, 1e-6);
}
