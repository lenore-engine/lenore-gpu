const std = @import("std");
const zm = @import("zmath");
const gpu = @import("lenore-gpu");

const testing = std.testing;
const Template = gpu.NodeTemplate;
const Animator = gpu.NodeAnimator;
const no_parent = gpu.no_parent;

const tolerance = 1e-5;

fn apply(matrix: zm.Mat, point: [3]f32) [3]f32 {
    const result = zm.mul(zm.f32x4(point[0], point[1], point[2], 1.0), matrix);
    return .{ result[0], result[1], result[2] };
}

fn expectPoint(expected: [3]f32, actual: [3]f32) !void {
    for (expected, actual) |want, got| try testing.expectApproxEqAbs(want, got, tolerance);
}

// A root that slides along x over two seconds, and a child hanging one unit
// above it that no channel targets.
const Built = struct {
    template: Template,

    fn deinit(self: *Built, allocator: std.mem.Allocator) void {
        self.template.deinit(allocator);
    }
};

fn build(allocator: std.mem.Allocator, target_slot: u32) !Built {
    const parent = try allocator.dupe(u16, &.{ no_parent, 0 });
    const prefix = try allocator.dupe(zm.Mat, &.{ zm.identity(), zm.identity() });
    const translations = try allocator.dupe(zm.Vec, &.{
        zm.f32x4(0, 0, 0, 0),
        zm.f32x4(0, 1, 0, 0),
    });
    const rotations = try allocator.dupe(zm.Quat, &.{ zm.qidentity(), zm.qidentity() });
    const scales = try allocator.dupe(zm.Vec, &.{
        zm.f32x4(1, 1, 1, 0),
        zm.f32x4(1, 1, 1, 0),
    });
    const bind_local = try allocator.dupe(zm.Mat, &.{
        zm.identity(),
        zm.translation(0, 1, 0),
    });

    const keys = try allocator.alloc(gpu.Keyframe(zm.Vec), 2);
    keys[0] = .{ .time = 0.0, .value = zm.f32x4(0, 0, 0, 0) };
    keys[1] = .{ .time = 2.0, .value = zm.f32x4(10, 0, 0, 0) };

    const channels = try allocator.alloc(gpu.AnimationChannel, 1);
    channels[0] = .{
        .track = .{ .translation = keys },
        .target_slot = target_slot,
        .interpolation = .linear,
    };

    const clips = try allocator.alloc(gpu.Animation, 1);
    clips[0] = try gpu.Animation.init(allocator, channels, "slide");

    const animated = try allocator.alloc([]u16, 1);
    animated[0] = try allocator.dupe(u16, &.{0});

    return .{ .template = try Template.init(allocator, .{
        .parent = parent,
        .prefix = prefix,
        .bind_translations = translations,
        .bind_rotations = rotations,
        .bind_scales = scales,
        .bind_local = bind_local,
        .clips = clips,
        .animated_slots = animated,
    }) };
}

test "an animator starts at the bind pose with valid world transforms" {
    var built = try build(testing.allocator, 0);
    defer built.deinit(testing.allocator);

    var animator = try Animator.init(testing.allocator, &built.template);
    defer animator.deinit(testing.allocator);

    // Nothing has played, and the world transforms are already usable.
    try expectPoint(.{ 0, 0, 0 }, apply(animator.world_transforms[0], .{ 0, 0, 0 }));
    try expectPoint(.{ 0, 1, 0 }, apply(animator.world_transforms[1], .{ 0, 0, 0 }));

    // An idle animator does no work and moves nothing.
    animator.update(1.0);
    try expectPoint(.{ 0, 0, 0 }, apply(animator.world_transforms[0], .{ 0, 0, 0 }));
}

test "motion of a parent reaches a child that no channel targets" {
    var built = try build(testing.allocator, 0);
    defer built.deinit(testing.allocator);

    var animator = try Animator.init(testing.allocator, &built.template);
    defer animator.deinit(testing.allocator);

    try animator.play(0);
    animator.update(1.0);

    try expectPoint(.{ 5, 0, 0 }, apply(animator.world_transforms[0], .{ 0, 0, 0 }));
    // The child keeps its bind local matrix and inherits the movement, which is
    // the case the per-clip animated slot list exists to keep cheap.
    try expectPoint(.{ 5, 1, 0 }, apply(animator.world_transforms[1], .{ 0, 0, 0 }));
}

test "stopping freezes the pose and playing again restarts it" {
    var built = try build(testing.allocator, 0);
    defer built.deinit(testing.allocator);

    var animator = try Animator.init(testing.allocator, &built.template);
    defer animator.deinit(testing.allocator);

    try animator.play(0);
    animator.update(1.0);
    animator.stop();

    animator.update(0.5);
    try expectPoint(.{ 5, 0, 0 }, apply(animator.world_transforms[0], .{ 0, 0, 0 }));

    try animator.play(0);
    animator.update(0.0);
    try expectPoint(.{ 0, 0, 0 }, apply(animator.world_transforms[0], .{ 0, 0, 0 }));
}

test "a clip index outside the document is refused" {
    var built = try build(testing.allocator, 0);
    defer built.deinit(testing.allocator);

    var animator = try Animator.init(testing.allocator, &built.template);
    defer animator.deinit(testing.allocator);

    try testing.expectError(error.NoSuchClip, animator.play(1));
    try testing.expect(animator.active_clip == null);
}

// The per-frame propagation indexes the parent array without a check, so the
// shapes that would need one are refused here.
test "a hierarchy whose channels reach past it is refused" {
    const allocator = testing.allocator;
    // A rejected document is destroyed by init, so nothing is left to free.
    try testing.expectError(error.ClipTargetOutOfRange, build(allocator, 5));
}

test "a parent that is not before its child is refused" {
    const allocator = testing.allocator;
    const parent = try allocator.dupe(u16, &.{ 1, no_parent });
    const prefix = try allocator.dupe(zm.Mat, &.{ zm.identity(), zm.identity() });
    const translations = try allocator.dupe(zm.Vec, &.{ zm.f32x4(0, 0, 0, 0), zm.f32x4(0, 0, 0, 0) });
    const rotations = try allocator.dupe(zm.Quat, &.{ zm.qidentity(), zm.qidentity() });
    const scales = try allocator.dupe(zm.Vec, &.{ zm.f32x4(1, 1, 1, 0), zm.f32x4(1, 1, 1, 0) });
    const bind_local = try allocator.dupe(zm.Mat, &.{ zm.identity(), zm.identity() });
    const clips = try allocator.alloc(gpu.Animation, 0);
    const animated = try allocator.alloc([]u16, 0);

    try testing.expectError(error.ParentNotBeforeChild, Template.init(allocator, .{
        .parent = parent,
        .prefix = prefix,
        .bind_translations = translations,
        .bind_rotations = rotations,
        .bind_scales = scales,
        .bind_local = bind_local,
        .clips = clips,
        .animated_slots = animated,
    }));
}
