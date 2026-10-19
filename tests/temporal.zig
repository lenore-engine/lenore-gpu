const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");
const zm = @import("zmath");

const testing = std.testing;
const temporal = gpu.Temporal;

// Device coordinates span two across the target, so a device offset of two
// over the width is one render pixel.
test "the jitter reaches the shader in render pixels" {
    const render: vk.Extent2D = .{ .width = 1920, .height = 1200 };
    const pixels = temporal.jitterPixels(.{ 2.0 / 1920.0 * 0.25, -2.0 / 1200.0 * 0.5 }, render);
    try testing.expectApproxEqAbs(@as(f32, 0.25), pixels[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -0.5), pixels[1], 1e-6);
}

test "the push block carries both extents, the reset and the sky" {
    const sky: temporal.Sky = .{ .across = .{ 1, 2, 3, 4 }, .down = .{ 5, 6, 7, 8 }, .ahead = .{ 9, 10, 11, 12 } };
    const push = temporal.pushFor(.{ .jitter = .{ 0, 0 }, .sky = sky }, .{ .width = 1920, .height = 1200 }, .{ .width = 2560, .height = 1600 }, true);
    try testing.expectEqual([2]f32{ 1920, 1200 }, push.render_size);
    try testing.expectEqual([2]f32{ 2560, 1600 }, push.output_size);
    try testing.expectEqual(@as(u32, 1), push.reset);
    try testing.expectEqual(sky.down, push.sky_down);
    try testing.expectEqual(@as(usize, 80), @sizeOf(temporal.PushConstants));
}

const fov_y: f32 = std.math.pi / 4.0;
const aspect: f32 = 1.6;

// A camera at `eye` turned `yaw` radians to the right of looking down -z, with
// the basis built the way the scene's `rayBasis` builds it, in the scene's
// clip space.
fn cameraAt(eye: [3]f32, yaw: f32) struct { view_projection: zm.Mat, ray: [3][4]f32 } {
    const front = zm.f32x4(@sin(yaw), 0, -@cos(yaw), 0);
    const view = zm.lookToRh(zm.f32x4(eye[0], eye[1], eye[2], 1), front, zm.f32x4(0, 1, 0, 0));
    const half_height = @tan(fov_y / 2.0);
    const half_width = half_height * aspect;
    return .{
        .view_projection = zm.mul(view, zm.perspectiveFovRh(fov_y, aspect, 0.1, 100)),
        .ray = .{
            .{ @cos(yaw) * half_width, 0, @sin(yaw) * half_width, 0 },
            .{ 0, half_height, 0, 0 },
            .{ front[0], front[1], front[2], 0 },
        },
    };
}

fn skyBetween(previous: anytype, current: anytype) temporal.Sky {
    return temporal.skyFor(gpu.vulkanClipCamera(.{
        .view_projection = gpu.mat4(current.view_projection),
        .previous_view_projection = gpu.mat4(previous.view_projection),
        .position = .{ 0, 0, 0, 0 },
        .ray_right = current.ray[0],
        .ray_up = current.ray[1],
        .ray_front = current.ray[2],
    }));
}

// Last frame's device position of the background at this frame's `device`,
// as the shader computes it.
fn previousDevice(sky: temporal.Sky, device: [2]f32) [2]f32 {
    var clip: [4]f32 = undefined;
    for (0..4) |lane| clip[lane] = device[0] * sky.across[lane] + device[1] * sky.down[lane] + sky.ahead[lane];
    return .{ clip[0] / clip[3], clip[1] / clip[3] };
}

const probes = [_][2]f32{ .{ 0, 0 }, .{ 0.7, -0.4 }, .{ -0.9, 0.95 } };

test "the background stays put when the camera does" {
    const camera = cameraAt(.{ 3, 1.7, -2 }, 0.3);
    const sky = skyBetween(camera, camera);
    for (probes) |device| {
        const before = previousDevice(sky, device);
        try testing.expectApproxEqAbs(device[0], before[0], 1e-5);
        try testing.expectApproxEqAbs(device[1], before[1], 1e-5);
    }
}

// The background is infinitely far, so walking does not move it on screen.
test "the camera's travel does not move the background" {
    const sky = skyBetween(cameraAt(.{ 0, 1.7, 0 }, 0.3), cameraAt(.{ 40, 3, -25 }, 0.3));
    for (probes) |device| {
        const before = previousDevice(sky, device);
        try testing.expectApproxEqAbs(device[0], before[0], 1e-5);
        try testing.expectApproxEqAbs(device[1], before[1], 1e-5);
    }
}

// Turning right brings in what was to the right: the centre's direction lay at
// `tan(turn)` across last frame's view, whose device x of one is
// `tan(fov_y / 2) * aspect` across at unit depth. Height is kept.
test "a turn moves the background by the angle turned" {
    const turn: f32 = 0.1;
    const sky = skyBetween(cameraAt(.{ 0, 1.7, 0 }, 0), cameraAt(.{ 0, 1.7, 0 }, turn));
    const before = previousDevice(sky, .{ 0, 0 });
    try testing.expectApproxEqAbs(@tan(turn) / (@tan(fov_y / 2.0) * aspect), before[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0), before[1], 1e-5);
}

// Each frame reads the history the frame before wrote, and never the one it
// is writing.
test "the history alternates and a frame reads what the last one wrote" {
    for (0..6) |frame| {
        const read = temporal.readIndex(frame);
        const written = 1 - read;
        try testing.expectEqual(written, temporal.readIndex(frame + 1));
    }
}

// The resolved picture and the history are read by bloom and the post pass in
// fragment shaders and by the meter in compute, so both scopes are named.
test "the resolve's outputs are handed to fragment and compute readers" {
    const done = temporal.colourBarrier(.null_handle, .attachment_to_sampled);
    try testing.expectEqual(gpu.mainPassSampledLayout, done.new_layout);
    try testing.expect(done.dst_stage_mask.fragment_shader_bit and done.dst_stage_mask.compute_shader_bit);
    const first = temporal.colourBarrier(.null_handle, .undefined_to_sampled);
    try testing.expectEqual(vk.ImageLayout.undefined, first.old_layout);
    try testing.expectEqual(gpu.mainPassSampledLayout, first.new_layout);
}
