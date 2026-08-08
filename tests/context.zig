const std = @import("std");
const gpu = @import("lenore-gpu");

const testing = std.testing;
const QueueSupport = gpu.QueueSupport;

// A family carrying everything, which is what an integrated part exposes first.
const universal: QueueSupport = .{ .graphics = true, .compute = true, .present = true };

test "one family serving both is preferred over a split pair" {
    // Preferred and not merely accepted: a shared family needs no ownership
    // transfer between the draw and the present. The split pair is offered
    // first here, so taking it would mean the scan stopped at the first match
    // rather than at the best one.
    const chosen = gpu.chooseQueues(&.{
        .{ .graphics = true, .compute = true, .present = false },
        .{ .graphics = false, .compute = false, .present = true },
        universal,
    }) orelse return error.TestExpectedQueues;

    try testing.expectEqual(@as(u32, 2), chosen.graphics_family);
    try testing.expectEqual(@as(u32, 2), chosen.present_family);
}

test "a split pair is taken when no family does both, and it is the first of each" {
    // Two candidates on each side, so the earliest is distinguishable from the
    // latest. A device exposes its most capable family first by convention and
    // nothing here ranks them further, which makes "the first that will do" the
    // whole rule rather than an accident of a one-candidate case.
    const chosen = gpu.chooseQueues(&.{
        .{ .graphics = false, .compute = true, .present = false },
        .{ .graphics = true, .compute = true, .present = false },
        .{ .graphics = false, .compute = false, .present = true },
        .{ .graphics = true, .compute = true, .present = false },
        .{ .graphics = false, .compute = false, .present = true },
    }) orelse return error.TestExpectedQueues;

    try testing.expectEqual(@as(u32, 1), chosen.graphics_family);
    try testing.expectEqual(@as(u32, 2), chosen.present_family);
}

test "a graphics family that cannot dispatch is not the graphics family" {
    // vk.xml gives vkCmdDispatch queues="VK_QUEUE_COMPUTE_BIT". The morph
    // prepass records into the same command buffer as the draws it feeds, so a
    // graphics-only family cannot carry the frame. Family one is what this
    // device is usable through, and taking family zero instead would be a
    // dispatch the layer refuses on every frame that has a morphed mesh.
    const chosen = gpu.chooseQueues(&.{
        .{ .graphics = true, .compute = false, .present = true },
        .{ .graphics = true, .compute = true, .present = true },
    }) orelse return error.TestExpectedQueues;

    try testing.expectEqual(@as(u32, 1), chosen.graphics_family);
    try testing.expectEqual(@as(u32, 1), chosen.present_family);
}

test "a device with no graphics-and-compute family is refused" {
    // Refused rather than fallen back on: without a dispatch the morph prepass
    // has nowhere to run, and a device chosen anyway would draw every morphed
    // mesh at its bind shape.
    try testing.expectEqual(@as(?gpu.QueueAllocation, null), gpu.chooseQueues(&.{
        .{ .graphics = true, .compute = false, .present = true },
        .{ .graphics = false, .compute = true, .present = false },
    }));
}

test "a device with no presentable family is refused" {
    try testing.expectEqual(@as(?gpu.QueueAllocation, null), gpu.chooseQueues(&.{
        .{ .graphics = true, .compute = true, .present = false },
    }));
}

test "a device exposing no queue families at all is refused" {
    try testing.expectEqual(@as(?gpu.QueueAllocation, null), gpu.chooseQueues(&.{}));
}
