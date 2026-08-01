const std = @import("std");
const zm = @import("zmath");
const gpu = @import("lenore-gpu");

const testing = std.testing;
const Template = gpu.SkeletonTemplate;
const Pose = gpu.SkeletonPose;
const no_parent = gpu.no_parent;

const tolerance = 1e-5;

// zmath composes for a row vector: a point is transformed as v * M, per its
// module header. Applying the matrices that way here is what makes these tests
// describe the convention the engine actually composes in, rather than the one a
// reader assumes.
fn apply(matrix: zm.Mat, point: [3]f32) [3]f32 {
    const result = zm.mul(zm.f32x4(point[0], point[1], point[2], 1.0), matrix);
    return .{ result[0], result[1], result[2] };
}

fn expectPoint(expected: [3]f32, actual: [3]f32) !void {
    for (expected, actual) |want, got| try testing.expectApproxEqAbs(want, got, tolerance);
}

// A root and one child hanging off it, with a single joint bound to the child.
const two_slot = struct {
    const parents = [_]u16{ no_parent, 0 };
    const prefixes = [_]zm.Mat{ zm.identity(), zm.identity() };
    const translations = [_]zm.Vec{
        zm.f32x4(1, 0, 0, 0),
        zm.f32x4(0, 2, 0, 0),
    };
    const rotations = [_]zm.Quat{ zm.qidentity(), zm.qidentity() };
    const scales = [_]zm.Vec{
        zm.f32x4(1, 1, 1, 0),
        zm.f32x4(1, 1, 1, 0),
    };
    const joint_slots = [_]u16{1};

    fn init(inverse_bind: []const zm.Mat) Template.Init {
        return .{
            .slot_parent = &parents,
            .slot_prefix = &prefixes,
            .bind_translations = &translations,
            .bind_rotations = &rotations,
            .bind_scales = &scales,
            .inverse_bind = inverse_bind,
            .joint_slot = &joint_slots,
        };
    }
};

test "a hierarchy accumulates parent transforms in one forward pass" {
    const inverse_bind = [_]zm.Mat{zm.identity()};
    var template = try Template.init(testing.allocator, two_slot.init(&inverse_bind));
    defer template.deinit(testing.allocator);

    var pose = try Pose.init(testing.allocator, &template);
    defer pose.deinit(testing.allocator);
    pose.evaluate();

    // The root moves by its own translation.
    try expectPoint(.{ 1, 0, 0 }, apply(pose.world_transforms[0], .{ 0, 0, 0 }));
    // The child carries its own translation and then its parent's.
    try expectPoint(.{ 1, 2, 0 }, apply(pose.world_transforms[1], .{ 0, 0, 0 }));
    // The joint reads the slot it is bound to, not its own index.
    try expectPoint(.{ 1, 2, 0 }, apply(pose.joint_transforms[0], .{ 0, 0, 0 }));
}

// The property the whole scheme rests on: with the true inverse bind matrices,
// an unposed skeleton leaves its vertices exactly where the asset put them.
test "the bind pose maps to identity through the inverse bind matrices" {
    var measuring = try Template.init(testing.allocator, two_slot.init(&.{zm.identity()}));
    var measuring_pose = try Pose.init(testing.allocator, &measuring);
    measuring_pose.evaluate();
    const bind_world = measuring_pose.world_transforms[1];
    measuring_pose.deinit(testing.allocator);
    measuring.deinit(testing.allocator);

    const inverse_bind = [_]zm.Mat{zm.inverse(bind_world)};
    var template = try Template.init(testing.allocator, two_slot.init(&inverse_bind));
    defer template.deinit(testing.allocator);

    var pose = try Pose.init(testing.allocator, &template);
    defer pose.deinit(testing.allocator);
    pose.evaluate();

    try expectPoint(.{ 0, 0, 0 }, apply(pose.joint_transforms[0], .{ 0, 0, 0 }));
    try expectPoint(.{ 3, -4, 5 }, apply(pose.joint_transforms[0], .{ 3, -4, 5 }));
}

test "a root is placed by its prefix" {
    const prefixes = [_]zm.Mat{ zm.translation(10, 0, 0), zm.identity() };
    const inverse_bind = [_]zm.Mat{zm.identity()};
    var params = two_slot.init(&inverse_bind);
    params.slot_prefix = &prefixes;

    var template = try Template.init(testing.allocator, params);
    defer template.deinit(testing.allocator);

    var pose = try Pose.init(testing.allocator, &template);
    defer pose.deinit(testing.allocator);
    pose.evaluate();

    try expectPoint(.{ 11, 0, 0 }, apply(pose.world_transforms[0], .{ 0, 0, 0 }));
    // The child inherits the placed root rather than the prefix of its own slot.
    try expectPoint(.{ 11, 2, 0 }, apply(pose.world_transforms[1], .{ 0, 0, 0 }));
}

test "posing writes locals and resetting restores the bind pose" {
    const inverse_bind = [_]zm.Mat{zm.identity()};
    var template = try Template.init(testing.allocator, two_slot.init(&inverse_bind));
    defer template.deinit(testing.allocator);

    var pose = try Pose.init(testing.allocator, &template);
    defer pose.deinit(testing.allocator);

    pose.local_translations[1] = zm.f32x4(0, 5, 0, 0);
    pose.evaluate();
    try expectPoint(.{ 1, 5, 0 }, apply(pose.joint_transforms[0], .{ 0, 0, 0 }));

    pose.resetToBindPose();
    pose.evaluate();
    try expectPoint(.{ 1, 2, 0 }, apply(pose.joint_transforms[0], .{ 0, 0, 0 }));
}

test "root motion displaces the first slot and its descendants" {
    const inverse_bind = [_]zm.Mat{zm.identity()};
    var template = try Template.init(testing.allocator, two_slot.init(&inverse_bind));
    defer template.deinit(testing.allocator);

    var pose = try Pose.init(testing.allocator, &template);
    defer pose.deinit(testing.allocator);

    pose.applyRootMotion(zm.f32x4(0, 0, 7, 0));
    pose.evaluate();

    try expectPoint(.{ 1, 0, 7 }, apply(pose.world_transforms[0], .{ 0, 0, 0 }));
    try expectPoint(.{ 1, 2, 7 }, apply(pose.world_transforms[1], .{ 0, 0, 0 }));
}

test "a skeleton with no slots is inert" {
    var template = try Template.init(testing.allocator, .{
        .slot_parent = &.{},
        .slot_prefix = &.{},
        .bind_translations = &.{},
        .bind_rotations = &.{},
        .bind_scales = &.{},
        .inverse_bind = &.{},
        .joint_slot = &.{},
    });
    defer template.deinit(testing.allocator);

    var pose = try Pose.init(testing.allocator, &template);
    defer pose.deinit(testing.allocator);

    pose.applyRootMotion(zm.f32x4(1, 1, 1, 0));
    pose.evaluate();
    try testing.expectEqual(@as(usize, 0), pose.jointCount());
}

// These come from an asset, and the per-frame pass has no check of its own
// because construction rejected the shapes that would need one.
test "malformed topology is rejected at construction" {
    const inverse_bind = [_]zm.Mat{zm.identity()};

    {
        // A parent that is not before its child would read a world transform
        // this frame has not written.
        const parents = [_]u16{ 1, no_parent };
        var params = two_slot.init(&inverse_bind);
        params.slot_parent = &parents;
        try testing.expectError(
            error.ParentNotBeforeChild,
            Template.init(testing.allocator, params),
        );
    }
    {
        // A slot naming itself is the same failure.
        const parents = [_]u16{ no_parent, 1 };
        var params = two_slot.init(&inverse_bind);
        params.slot_parent = &parents;
        try testing.expectError(
            error.ParentNotBeforeChild,
            Template.init(testing.allocator, params),
        );
    }
    {
        // A joint pointing outside the hierarchy would read outside the array.
        const slots = [_]u16{2};
        var params = two_slot.init(&inverse_bind);
        params.joint_slot = &slots;
        try testing.expectError(
            error.JointSlotOutOfRange,
            Template.init(testing.allocator, params),
        );
    }
    {
        const scales = [_]zm.Vec{zm.f32x4(1, 1, 1, 0)};
        var params = two_slot.init(&inverse_bind);
        params.bind_scales = &scales;
        try testing.expectError(
            error.SlotCountMismatch,
            Template.init(testing.allocator, params),
        );
    }
    {
        const slots = [_]u16{ 1, 1 };
        var params = two_slot.init(&inverse_bind);
        params.joint_slot = &slots;
        try testing.expectError(
            error.SlotCountMismatch,
            Template.init(testing.allocator, params),
        );
    }
}

test "the slot index space is bounded by what a parent reference can address" {
    const too_many = try testing.allocator.alloc(u16, std.math.maxInt(u16) + 2);
    defer testing.allocator.free(too_many);
    @memset(too_many, no_parent);

    try testing.expectError(error.TooManySlots, Template.init(testing.allocator, .{
        .slot_parent = too_many,
        .slot_prefix = &.{},
        .bind_translations = &.{},
        .bind_rotations = &.{},
        .bind_scales = &.{},
        .inverse_bind = &.{},
        .joint_slot = &.{},
    }));
}

test "composition applies scale, then rotation, then translation" {
    const quarter_turn = zm.quatFromAxisAngle(zm.f32x4(0, 0, 1, 0), std.math.pi / 2.0);
    const matrix = gpu.composeTransform(
        zm.f32x4(10, 0, 0, 0),
        quarter_turn,
        zm.f32x4(2, 2, 2, 0),
    );

    // A unit x point scales to (2,0,0), turns about z to (0,2,0), then
    // translates to (10,2,0). Any other order gives a different point.
    try expectPoint(.{ 10, 2, 0 }, apply(matrix, .{ 1, 0, 0 }));
}
