const std = @import("std");
const vk = @import("vulkan");
const zm = @import("zmath");
const gpu = @import("lenore-gpu");

const testing = std.testing;

// A camera the capacity checks below never read. Named once so that a block
// which grows does not spread through tests whose subject is the ring bounds.
//
// It goes through the flip like any other camera, because that is the only way
// to obtain what a frame's camera field holds.
const any_camera: gpu.FramebufferCamera = gpu.vulkanClipCamera(.{
    .view_projection = gpu.mat4(zm.identity()),
    .position = .{ 0, 0, 0, 1 },
    .ray_right = .{ 1, 0, 0, 0 },
    .ray_up = .{ 0, 1, 0, 0 },
    .ray_front = .{ 0, 0, -1, 0 },
    .previous_view_projection = gpu.mat4(zm.identity()),
});

test "the frame set's own methods are reached by the compiler" {
    // Every one needs a device to call.
    _ = &gpu.FrameSet.init;
    _ = &gpu.FrameSet.deinit;
    _ = &gpu.FrameSet.update;
    _ = &gpu.FrameSet.dynamicOffsets;
    _ = &gpu.FrameSet.bind;
    _ = &gpu.FrameSet.descriptorSetLayout;
}

test "an instance carries model, joint and material indices at the shader's stride" {
    // The element the shader steps through. A `float4x4` aligns the struct to
    // sixteen bytes: two matrices and four words stride by a hundred and
    // forty-four. Getting this wrong reads the wrong instance from the second
    // one on, and the first still looks right, which is why it is pinned here
    // rather than noticed.
    try testing.expectEqual(@as(usize, 144), @sizeOf(gpu.Instance));
    try testing.expectEqual(@as(usize, 16), @alignOf(gpu.Instance));

    // The indices sit after the matrix in the order the shader names them. A
    // different order has the same stride and still reads different values.
    try testing.expectEqual(@as(usize, 0), @offsetOf(gpu.Instance, "model"));
    try testing.expectEqual(@as(usize, 64), @offsetOf(gpu.Instance, "joint_base"));
    try testing.expectEqual(@as(usize, 68), @offsetOf(gpu.Instance, "material_index"));
    // After the first eighty bytes, which keep the place they had for a shader
    // that reads only those.
    try testing.expectEqual(@as(usize, 80), @offsetOf(gpu.Instance, "previous_model"));

    const default_material: gpu.Instance = .{ .model = gpu.mat4(zm.identity()), .joint_base = 0, .previous_model = gpu.mat4(zm.identity()) };
    try testing.expectEqual(@as(u32, 0), default_material.material_index);
}

test "a joint matrix is the matrix the shader reads and nothing more" {
    // `StructuredBuffer<float4x4>` has a sixty-four byte stride with no padding,
    // so the ring's element is exactly that and its stride is the shader's.
    try testing.expectEqual(@as(usize, 64), @sizeOf(gpu.Joint));
}

test "the frame bindings are distinct slots in one set" {
    for (gpu.FrameSetBindings, 0..) |binding, index| {
        try testing.expectEqual(@as(u32, 1), binding.count);
        // A binding no stage names is one the layout carries for nothing, and
        // nothing on the device objects to it.
        try testing.expect(binding.stages.toInt() != 0);
        for (gpu.FrameSetBindings[index + 1 ..]) |other| {
            try testing.expect(binding.slot != other.slot);
        }
    }
}

test "the lights reach the stage that shades and the matrices the one that transforms" {
    // Which stage a binding names is not bookkeeping: a block declared
    // vertex-only and read by the fragment shader is a layout the shader
    // disagrees with, and what comes out is black rather than wrong.
    try testing.expect(bindingFor(0).stages.vertex_bit);
    try testing.expect(bindingFor(1).stages.vertex_bit);
    try testing.expect(bindingFor(2).stages.fragment_bit);
    try testing.expect(!bindingFor(2).stages.vertex_bit);
    // The joints are read where the position is built, not where it is shaded.
    try testing.expect(bindingFor(3).stages.vertex_bit);
    try testing.expect(!bindingFor(3).stages.fragment_bit);
}

test "the rings that a scene sizes are storage buffers, not uniform ones" {
    // The camera and the lights are fixed blocks and fit the sixteen kilobytes
    // `maxUniformBufferRange` guarantees. The instances and the joints are sized
    // by the scene, and at sixty-four and eighty bytes an element that bound is
    // a couple of hundred of either for the whole frame.
    try testing.expectEqual(vk.DescriptorType.uniform_buffer_dynamic, bindingFor(0).kind);
    try testing.expectEqual(vk.DescriptorType.storage_buffer_dynamic, bindingFor(1).kind);
    try testing.expectEqual(vk.DescriptorType.uniform_buffer_dynamic, bindingFor(2).kind);
    try testing.expectEqual(vk.DescriptorType.storage_buffer_dynamic, bindingFor(3).kind);
}

test "a frame that fits is accepted at every bound and one past any of them is not" {
    const models: [4]gpu.Instance = @splat(.{ .model = gpu.mat4(zm.identity()), .joint_base = 0, .previous_model = gpu.mat4(zm.identity()) });
    const joints: [6]gpu.Joint = @splat(gpu.mat4(zm.identity()));
    var lights: [gpu.max_lights + 1]gpu.LightUniform = @splat(gpu.LightUniform.directional(.{ 1, 1, 1 }, 1, .{ 0, -1, 0 }));

    // Exactly full on every axis. The bound is the last index that fits, so a
    // check written with the wrong comparison rejects this and nothing else.
    try gpu.validateFrameContents(.{ .instances = 4, .joints = 6 }, .{
        .camera = any_camera,
        .models = &models,
        .joints = &joints,
        .previous_joints = &joints,
        .lights = lights[0..gpu.max_lights],
    });

    try testing.expectError(error.InstanceCapacityExceeded, gpu.validateFrameContents(.{ .instances = 3, .joints = 6 }, .{
        .camera = any_camera,
        .models = &models,
        .joints = &joints,
        .previous_joints = &joints,
        .lights = &.{},
    }));
    // The joint bound is its own: a frame can fit every instance and still
    // carry more matrices than the ring holds, since one instance's skeleton
    // decides how many it brings.
    try testing.expectError(error.JointCapacityExceeded, gpu.validateFrameContents(.{ .instances = 4, .joints = 5 }, .{
        .camera = any_camera,
        .models = &models,
        .joints = &joints,
        .previous_joints = &joints,
        .lights = &.{},
    }));
    try testing.expectError(error.LightCapacityExceeded, gpu.validateFrameContents(.{ .instances = 4, .joints = 6 }, .{
        .camera = any_camera,
        .models = &models,
        .joints = &joints,
        .previous_joints = &joints,
        .lights = &lights,
    }));
}

test "a frame with no skinned instance carries no joints" {
    // The unskinned case is the common one and it must not need a ring: a
    // capacity of zero accepts an empty joint slice.
    try gpu.validateFrameContents(.{ .instances = 1, .joints = 0 }, .{
        .camera = any_camera,
        .models = &.{.{ .model = gpu.mat4(zm.identity()), .joint_base = 0, .previous_model = gpu.mat4(zm.identity()) }},
        .joints = &.{},
        .lights = &.{},
    });
}

fn bindingFor(slot: u32) gpu.DescriptorBinding {
    for (gpu.FrameSetBindings) |binding| {
        if (binding.slot == slot) return binding;
    }
    unreachable;
}

// The shader indexes the shadow matrices by a light's layer with no check of its
// own, so the frame is where a layer that runs past them is stopped.
test "a light's shadow layers must lie inside the frame's matrices" {
    const capacity: gpu.FrameCapacity = .{ .instances = 1, .joints = 0 };
    const identity = gpu.mat4(zm.identity());
    const seven: [7]gpu.Mat4 = @splat(identity);

    var spot = gpu.LightUniform.spot(.{ 1, 1, 1 }, 5, .{
        .position = .{ 0, 3, 0 },
        .direction = .{ 0, -1, 0 },
        .range = 8,
        .cos_inner = 0.9,
        .cos_outer = 0.8,
    });
    var lamp = gpu.LightUniform.point(.{ 1, 1, 1 }, 5, .{ 0, 2, 0 }, 8);
    lamp.shadow_layer = 0;
    spot.shadow_layer = 6;
    try gpu.validateFrameContents(capacity, .{
        .camera = any_camera,
        .models = &.{},
        .joints = &.{},
        .lights = &.{ lamp, spot },
        .shadow_view_projections = &seven,
    });

    // A point light's six layers end one past the seven matrices.
    lamp.shadow_layer = 2;
    try testing.expectError(error.ShadowLayerOutOfRange, gpu.validateFrameContents(capacity, .{
        .camera = any_camera,
        .models = &.{},
        .joints = &.{},
        .lights = &.{lamp},
        .shadow_view_projections = &seven,
    }));

    // Near the top of u32, where a sum would wrap below the count.
    spot.shadow_layer = std.math.maxInt(u32) - 1;
    try testing.expectError(error.ShadowLayerOutOfRange, gpu.validateFrameContents(capacity, .{
        .camera = any_camera,
        .models = &.{},
        .joints = &.{},
        .lights = &.{spot},
        .shadow_view_projections = &seven,
    }));

    // A kind with no local map.
    var sun = gpu.LightUniform.directional(.{ 1, 1, 1 }, 3, .{ 0, -1, 0 });
    sun.shadow_layer = 0;
    try testing.expectError(error.ShadowLayerOutOfRange, gpu.validateFrameContents(capacity, .{
        .camera = any_camera,
        .models = &.{},
        .joints = &.{},
        .lights = &.{sun},
        .shadow_view_projections = &seven,
    }));

    const too_many: [gpu.max_shadow_layers + 1]gpu.Mat4 = @splat(identity);
    try testing.expectError(error.ShadowLayerCapacityExceeded, gpu.validateFrameContents(capacity, .{
        .camera = any_camera,
        .models = &.{},
        .joints = &.{},
        .lights = &.{},
        .shadow_view_projections = &too_many,
    }));
}

test "previous joints must match the current pose length" {
    const joints: [2]gpu.Joint = @splat(gpu.mat4(zm.identity()));
    var frame: gpu.FrameContents = .{
        .camera = any_camera,
        .models = &.{},
        .joints = &joints,
        .previous_joints = &joints,
        .lights = &.{},
    };
    const capacity: gpu.FrameCapacity = .{ .instances = 0, .joints = 2 };
    try gpu.validateFrameContents(capacity, frame);
    frame.previous_joints = joints[0..1];
    try testing.expectError(error.PreviousJointCountMismatch, gpu.validateFrameContents(capacity, frame));
    frame.joints = &.{};
    try testing.expectError(error.PreviousJointCountMismatch, gpu.validateFrameContents(capacity, frame));
    frame.previous_joints = &.{};
    try gpu.validateFrameContents(capacity, frame);
}

test "joint poses occupy consecutive live runs within a doubled ring" {
    const sentinel = gpu.mat4(zm.translation(-1, -1, -1));
    var storage: [8]gpu.Joint = @splat(sentinel);
    const current = [_]gpu.Joint{ gpu.mat4(zm.translation(1, 0, 0)), gpu.mat4(zm.translation(2, 0, 0)) };
    const previous = [_]gpu.Joint{ gpu.mat4(zm.translation(3, 0, 0)), gpu.mat4(zm.translation(4, 0, 0)) };
    gpu.writeJointPoses(&storage, &current, &previous);
    try testing.expectEqualSlices(gpu.Joint, &current, storage[0..2]);
    try testing.expectEqualSlices(gpu.Joint, &previous, storage[2..4]);
    for (storage[4..]) |joint| try testing.expectEqual(sentinel, joint);
    gpu.writeJointPoses(&storage, &.{}, &.{});
    try testing.expectEqualSlices(gpu.Joint, &previous, storage[2..4]);

    var frame_set: gpu.FrameSet = undefined;
    frame_set.instances.count = 3;
    frame_set.joints.count = storage.len;
    try testing.expectEqual(gpu.FrameCapacity{ .instances = 3, .joints = 4 }, frame_set.capacity());
}
