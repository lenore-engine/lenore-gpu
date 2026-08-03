const std = @import("std");
const vk = @import("vulkan");
const zm = @import("zmath");
const gpu = @import("lenore-gpu");

const testing = std.testing;

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
    // sixteen bytes, so seventy-two bytes of payload stride by eighty. Getting
    // this wrong reads the wrong instance from the second one on, and the first
    // still looks right, which is why it is pinned here rather than noticed.
    try testing.expectEqual(@as(usize, 80), @sizeOf(gpu.Instance));
    try testing.expectEqual(@as(usize, 16), @alignOf(gpu.Instance));

    // The indices sit after the matrix in the order the shader names them. A
    // different order has the same stride and still reads different values.
    try testing.expectEqual(@as(usize, 0), @offsetOf(gpu.Instance, "model"));
    try testing.expectEqual(@as(usize, 64), @offsetOf(gpu.Instance, "joint_base"));
    try testing.expectEqual(@as(usize, 68), @offsetOf(gpu.Instance, "material_index"));

    const default_material: gpu.Instance = .{ .model = zm.identity(), .joint_base = 0 };
    try testing.expectEqual(@as(u32, 0), default_material.material_index);
}

test "a joint matrix is the matrix the shader reads and nothing more" {
    // `StructuredBuffer<float4x4>` has a sixty-four byte stride with no padding.
    // A pose is copied into the ring whole, so this is also what makes that copy
    // a `memcpy` rather than a gather.
    try testing.expectEqual(@as(usize, 64), @sizeOf(gpu.Joint));
    try testing.expectEqual(@as(usize, 16), @alignOf(gpu.Joint));
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
    const models = [_]gpu.Instance{.{ .model = zm.identity(), .joint_base = 0 }} ** 4;
    const joints = [_]gpu.Joint{zm.identity()} ** 6;
    var lights = [_]gpu.LightUniform{gpu.LightUniform.directional(.{ 1, 1, 1 }, 1, .{ 0, -1, 0 })} ** (gpu.max_lights + 1);
    const camera: gpu.CameraUniform = .{
        .view_projection = zm.identity(),
        .position = .{ 0, 0, 0, 1 },
    };

    // Exactly full on every axis. The bound is the last index that fits, so a
    // check written with the wrong comparison rejects this and nothing else.
    try gpu.validateFrameContents(.{ .instances = 4, .joints = 6 }, .{
        .camera = camera,
        .models = &models,
        .joints = &joints,
        .lights = lights[0..gpu.max_lights],
    });

    try testing.expectError(error.InstanceCapacityExceeded, gpu.validateFrameContents(.{ .instances = 3, .joints = 6 }, .{
        .camera = camera,
        .models = &models,
        .joints = &joints,
        .lights = &.{},
    }));
    // The joint bound is its own: a frame can fit every instance and still
    // carry more matrices than the ring holds, since one instance's skeleton
    // decides how many it brings.
    try testing.expectError(error.JointCapacityExceeded, gpu.validateFrameContents(.{ .instances = 4, .joints = 5 }, .{
        .camera = camera,
        .models = &models,
        .joints = &joints,
        .lights = &.{},
    }));
    try testing.expectError(error.LightCapacityExceeded, gpu.validateFrameContents(.{ .instances = 4, .joints = 6 }, .{
        .camera = camera,
        .models = &models,
        .joints = &joints,
        .lights = &lights,
    }));
}

test "a frame with no skinned instance carries no joints" {
    // The unskinned case is the common one and it must not need a ring: a
    // capacity of zero accepts an empty joint slice.
    try gpu.validateFrameContents(.{ .instances = 1, .joints = 0 }, .{
        .camera = .{ .view_projection = zm.identity(), .position = .{ 0, 0, 0, 1 } },
        .models = &.{.{ .model = zm.identity(), .joint_base = 0 }},
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
