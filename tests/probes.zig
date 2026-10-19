const std = @import("std");
const gpu = @import("lenore-gpu");

const testing = std.testing;
const probes = gpu.Probes;

// Measured against the compiler's reflection of probes.slang in the umbrella's
// shader_reflection.zig; restated here so a change fails without the shader.
test "the update's records are laid out as the shader reads them" {
    try testing.expectEqual(@as(usize, 32), @sizeOf(probes.GridParams));
    try testing.expectEqual(@as(usize, 48), @sizeOf(probes.Triangle));
    try testing.expectEqual(@as(usize, 96), @sizeOf(probes.PushConstants));
    try testing.expectEqual(@as(usize, 32), @offsetOf(probes.PushConstants, "sky"));
    try testing.expectEqual(@as(usize, 64), @offsetOf(probes.PushConstants, "rotation"));
    try testing.expectEqual(@as(usize, 88), @offsetOf(probes.PushConstants, "max_distance"));
}

test "update rotations are unit quaternions that do not repeat soon" {
    var previous: [4]f32 = .{ 0, 0, 0, 0 };
    for (0..64) |i| {
        const q = probes.rotationFor(@intCast(i));
        const length = @sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        try testing.expectApproxEqAbs(@as(f32, 1), length, 1e-5);
        if (i > 0) {
            const dot = @abs(q[0] * previous[0] + q[1] * previous[1] + q[2] * previous[2] + q[3] * previous[3]);
            try testing.expect(dot < 0.9999);
        }
        previous = q;
    }
}

test "a grid with no probes is no field" {
    try testing.expectEqual(@as(u32, 0), probes.GridParams.none.probeCount());
    try testing.expectEqual(@as(u32, 0), probes.GridParams.none.enabled);
}

test "settling averages the first updates, then keeps a fixed share" {
    try testing.expectEqual(@as(f32, 0), probes.settleHysteresis(0, 0.98));
    try testing.expectApproxEqAbs(@as(f32, 0.5), probes.settleHysteresis(1, 0.98), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.9), probes.settleHysteresis(9, 0.98), 1e-6);
    try testing.expectEqual(@as(f32, 0.98), probes.settleHysteresis(200, 0.98));
}
