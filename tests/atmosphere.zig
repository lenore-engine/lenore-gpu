const std = @import("std");
const gpu = @import("lenore-gpu");
const testing = std.testing;

test "Vulkan cube face centres and all corners" {
    const centres = [_][3]f32{ .{ 1, 0, 0 }, .{ -1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, -1, 0 }, .{ 0, 0, 1 }, .{ 0, 0, -1 } };
    // Independent inverse checks against the Vulkan coordinate selection table:
    // (sc,tc) = (-z,-y), (z,-y), (x,z), (x,-z), (x,-y), (-x,-y).
    for (centres, 0..) |centre, face| {
        const d = try gpu.cubeDirection(@intCast(face), .{ 0.5, 0.5 });
        for (0..3) |c| try testing.expectApproxEqAbs(centre[c], d[c], 0.000001);
        for ([_]f32{ 0, 1 }) |u| for ([_]f32{ 0, 1 }) |v| {
            const corner = try gpu.cubeDirection(@intCast(face), .{ u, v });
            const sc: f32 = switch (face) {
                0 => -corner[2],
                1 => corner[2],
                2, 3, 4 => corner[0],
                else => -corner[0],
            };
            const tc: f32 = switch (face) {
                2 => corner[2],
                3 => -corner[2],
                else => -corner[1],
            };
            const major = @abs(corner[face / 2]);
            try testing.expectApproxEqAbs(2 * u - 1, sc / major, 0.000001);
            try testing.expectApproxEqAbs(2 * v - 1, tc / major, 0.000001);
        };
    }
    try testing.expectError(error.InvalidCubeCoordinate, gpu.cubeDirection(6, .{ 0.5, 0.5 }));
    try testing.expectError(error.InvalidCubeCoordinate, gpu.cubeDirection(0, .{ std.math.nan(f32), 0 }));
}

test "ambient keeps the shader block offsets and enables reflected ambient by default" {
    try testing.expectEqual(@as(usize, 32), @sizeOf(gpu.AmbientUniform));
    try testing.expectEqual(@as(usize, 16), @alignOf(gpu.AmbientUniform));
    try testing.expectEqual(@as(usize, 0), @offsetOf(gpu.AmbientUniform, "sky"));
    try testing.expectEqual(@as(usize, 12), @offsetOf(gpu.AmbientUniform, "reflected"));
    try testing.expectEqual(@as(usize, 16), @offsetOf(gpu.AmbientUniform, "ground"));
    try testing.expectEqual(@as(usize, 28), @offsetOf(gpu.AmbientUniform, "padding_1"));
    const ambient: gpu.AmbientUniform = .{};
    try testing.expectEqual(@as(f32, 1), ambient.reflected);
    try testing.expectEqual(@as(usize, 144), @sizeOf(gpu.AtmosphereParameters));
    try testing.expectEqual(@as(usize, 16), @sizeOf(gpu.AtmosphereSun));
}
