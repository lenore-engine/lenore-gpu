const std = @import("std");
const gpu = @import("lenore-gpu");
const zm = @import("zmath");

const testing = std.testing;

test "a light kind fills only the lanes it owns" {
    // The whole reason the record is not the reference's four tagged vec4s. A
    // directional light has no position and no range, so reading either has to
    // give zero rather than whatever the last light of another kind left there.
    const sun = gpu.LightUniform.directional(.{ 1, 0.9, 0.8 }, 3, .{ 0, -1, 0 });
    try testing.expectEqual(gpu.LightUniform.Kind.directional, sun.kind);
    try testing.expectEqual([3]f32{ 0, 0, 0 }, sun.position);
    try testing.expectEqual(@as(f32, 0), sun.range);
    try testing.expectEqual(@as(f32, 0), sun.cos_inner);
    try testing.expectEqual(@as(f32, 0), sun.cos_outer);

    const bulb = gpu.LightUniform.point(.{ 1, 1, 1 }, 12, .{ 2, 3, 4 }, 10);
    try testing.expectEqual(gpu.LightUniform.Kind.point, bulb.kind);
    try testing.expectEqual([3]f32{ 0, 0, 0 }, bulb.direction);
    try testing.expectEqual(@as(f32, 0), bulb.cos_outer);

    const torch = gpu.LightUniform.spot(.{ 1, 1, 1 }, 12, .{
        .position = .{ 2, 3, 4 },
        .direction = .{ 0, 0, -1 },
        .range = 10,
        .cos_inner = 0.9,
        .cos_outer = 0.7,
    });
    try testing.expectEqual(gpu.LightUniform.Kind.spot, torch.kind);
    try testing.expect(torch.cos_inner > torch.cos_outer);
}

test "the light record is laid out as the shader's array steps through it" {
    // Measured against the compiler's reflection in `reflection.zig`; this is
    // the same statement without the shader, so a change to the record fails
    // here even when the shaders are not rebuilt.
    try testing.expectEqual(@as(usize, 64), @sizeOf(gpu.LightUniform));
    try testing.expectEqual(@as(usize, 16), @alignOf(gpu.LightUniform));
    try testing.expectEqual(@as(usize, 12), @offsetOf(gpu.LightUniform, "kind"));
    try testing.expectEqual(@as(usize, 28), @offsetOf(gpu.LightUniform, "intensity"));
    try testing.expectEqual(@as(usize, 44), @offsetOf(gpu.LightUniform, "range"));
}

test "the lights block begins its array where the shader's does" {
    // A count occupying four bytes and an array wanting sixteen is exactly the
    // case a hand-written mirror gets wrong, and the array then reads one
    // light's worth of the previous one's tail.
    try testing.expectEqual(@as(usize, 16), @offsetOf(gpu.LightsUniform, "lights"));
    try testing.expectEqual(
        @as(usize, 16 + gpu.max_lights * 64),
        @sizeOf(gpu.LightsUniform),
    );
}

test "filling the block states how much of it is live and leaves the rest alone" {
    var block: gpu.LightsUniform = undefined;
    const sun = gpu.LightUniform.directional(.{ 1, 1, 1 }, 3, .{ 0, -1, 0 });
    const lamp = gpu.LightUniform.point(.{ 0, 1, 0 }, 5, .{ 1, 2, 3 }, 8);

    block.fill(&.{ sun, lamp });
    try testing.expectEqual(@as(u32, 2), block.count[0]);
    try testing.expectEqual(sun, block.lights[0]);
    try testing.expectEqual(lamp, block.lights[1]);

    // A shorter frame after a longer one: the count is what the shader stops
    // at, so the light left behind at index one has to become unreachable
    // rather than be cleared.
    block.fill(&.{lamp});
    try testing.expectEqual(@as(u32, 1), block.count[0]);
    try testing.expectEqual(lamp, block.lights[0]);
    try testing.expectEqual(lamp, block.lights[1]);

    block.fill(&.{});
    try testing.expectEqual(@as(u32, 0), block.count[0]);
}

test "the block's kind values are the ones the shader branches on" {
    // The shader compares against literal 0, 1 and 2, which is what a uint tag
    // reduces to. Reordering the enum here would re-aim every light.
    try testing.expectEqual(@as(u32, 0), @intFromEnum(gpu.LightUniform.Kind.directional));
    try testing.expectEqual(@as(u32, 1), @intFromEnum(gpu.LightUniform.Kind.point));
    try testing.expectEqual(@as(u32, 2), @intFromEnum(gpu.LightUniform.Kind.spot));
}

test "the flip negates the whole second column, not one entry" {
    // A view-projection is a product, so its second column is not the sparse
    // one a bare perspective matrix has. Negating only the diagonal would leave
    // the other three terms carrying the old sign.
    const matrix: zm.Mat = .{
        zm.f32x4(1, 2, 3, 4),
        zm.f32x4(5, 6, 7, 8),
        zm.f32x4(9, 10, 11, 12),
        zm.f32x4(13, 14, 15, 16),
    };
    const flipped = gpu.vulkanClip(matrix);

    inline for (0..4) |row| {
        inline for (0..4) |column| {
            const expected = if (column == 1) -matrix[row][column] else matrix[row][column];
            try testing.expectEqual(expected, flipped[row][column]);
        }
    }
}

test "flipping twice is the identity" {
    const matrix = zm.perspectiveFovRh(std.math.pi / 4.0, 1.7778, 0.1, 100);
    const twice = gpu.vulkanClip(gpu.vulkanClip(matrix));
    inline for (0..4) |row| {
        inline for (0..4) |column| {
            try testing.expectEqual(matrix[row][column], twice[row][column]);
        }
    }
}

test "a point above the eye lands below the centre of the framebuffer" {
    // The whole point of the flip, stated as the thing it changes. zmath's
    // right-handed builders put the eye at the origin looking down -Z, so a
    // point at +Y is above it.
    const view = zm.lookToRh(zm.f32x4(0, 0, 0, 1), zm.f32x4(0, 0, -1, 0), zm.f32x4(0, 1, 0, 0));
    const projection = zm.perspectiveFovRh(std.math.pi / 4.0, 1, 0.1, 100);
    const view_projection = zm.mul(view, projection);
    const above = zm.f32x4(0, 1, -5, 1);

    const unflipped = zm.mul(above, view_projection);
    try testing.expect(unflipped[1] / unflipped[3] > 0);

    // Device y grows downward once flipped, so the same point is negative,
    // which is the upper half of the window.
    const flipped = zm.mul(above, gpu.vulkanClip(view_projection));
    try testing.expect(flipped[1] / flipped[3] < 0);

    // Nothing else moves: x and depth are untouched.
    try testing.expectEqual(unflipped[0], flipped[0]);
    try testing.expectEqual(unflipped[2], flipped[2]);
    try testing.expectEqual(unflipped[3], flipped[3]);
}
