const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;

test "a dependency reaches the barrier without crossing its own scopes" {
    // Four masks that no two positions share, so a source written into a
    // destination member or a stage into an access member shows up as a value
    // that could not have come from anywhere else. The subject is the mapping
    // and nothing else: a swapped pair still produces a valid barrier, orders
    // the opposite of what was asked, and is reported by no layer.
    const barrier = gpu.memoryBarrier(.{
        .src_stage = .{ .compute_shader_bit = true },
        .src_access = .{ .shader_storage_write_bit = true },
        .dst_stage = .{ .vertex_attribute_input_bit = true },
        .dst_access = .{ .vertex_attribute_read_bit = true },
    });

    try testing.expectEqual(
        vk.PipelineStageFlags2{ .compute_shader_bit = true },
        barrier.src_stage_mask,
    );
    try testing.expectEqual(
        vk.AccessFlags2{ .shader_storage_write_bit = true },
        barrier.src_access_mask,
    );
    try testing.expectEqual(
        vk.PipelineStageFlags2{ .vertex_attribute_input_bit = true },
        barrier.dst_stage_mask,
    );
    try testing.expectEqual(
        vk.AccessFlags2{ .vertex_attribute_read_bit = true },
        barrier.dst_access_mask,
    );
}

test "a destination scope of several stages survives as one mask" {
    // What a dispatch whose output is read three ways needs, and the case a set
    // of named dependencies could not have expressed: the mask is a union, not
    // a choice between cases.
    const barrier = gpu.memoryBarrier(.{
        .src_stage = .{ .compute_shader_bit = true },
        .src_access = .{ .shader_storage_write_bit = true },
        .dst_stage = .{ .copy_bit = true, .compute_shader_bit = true, .fragment_shader_bit = true },
        .dst_access = .{ .transfer_read_bit = true, .shader_storage_read_bit = true },
    });

    try testing.expect(barrier.dst_stage_mask.copy_bit);
    try testing.expect(barrier.dst_stage_mask.compute_shader_bit);
    try testing.expect(barrier.dst_stage_mask.fragment_shader_bit);
    try testing.expect(!barrier.dst_stage_mask.vertex_shader_bit);

    try testing.expect(barrier.dst_access_mask.transfer_read_bit);
    try testing.expect(barrier.dst_access_mask.shader_storage_read_bit);
    try testing.expect(!barrier.dst_access_mask.shader_storage_write_bit);
}

test "the prepass dependency survives the trip through a barrier" {
    // The morph prepass states its dependency in this vocabulary and records it
    // through `recordMemoryBarrier`, so what the device sees is this value after
    // one mapping. Held here rather than in the morph tests because what could
    // go wrong is the mapping.
    const declared = gpu.morphDependency();
    const barrier = gpu.memoryBarrier(declared);

    try testing.expectEqual(declared.src_stage, barrier.src_stage_mask);
    try testing.expectEqual(declared.src_access, barrier.src_access_mask);
    try testing.expectEqual(declared.dst_stage, barrier.dst_stage_mask);
    try testing.expectEqual(declared.dst_access, barrier.dst_access_mask);
    try testing.expect(barrier.dst_stage_mask.vertex_attribute_input_bit);
    try testing.expect(!barrier.dst_stage_mask.vertex_shader_bit);
}

test "recording a barrier needs a device and is reached by the compiler" {
    _ = &gpu.recordMemoryBarrier;
}
