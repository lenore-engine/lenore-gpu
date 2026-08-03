const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;

const vertex: vk.ShaderStageFlags = .{ .vertex_bit = true };
const fragment: vk.ShaderStageFlags = .{ .fragment_bit = true };

// The set the first lit frame needs, as the shader will declare it: the
// per-frame buffers carry a dynamic offset so that one set serves every frame.
const frame_set = [_]gpu.DescriptorBinding{
    .{ .slot = 0, .kind = .uniform_buffer_dynamic, .stages = vertex },
    .{ .slot = 1, .kind = .storage_buffer_dynamic, .stages = vertex },
    .{ .slot = 2, .kind = .uniform_buffer_dynamic, .stages = fragment },
    .{ .slot = 3, .kind = .storage_buffer, .stages = fragment },
};

test "descriptors of one type are summed into a single pool entry" {
    const sizes = gpu.DescriptorSets(&frame_set).pool_sizes_per_set;

    // Four bindings, three types: the two dynamic uniform buffers merge.
    try testing.expectEqual(@as(usize, 3), sizes.len);
    try testing.expectEqual(vk.DescriptorType.uniform_buffer_dynamic, sizes[0].type);
    try testing.expectEqual(@as(u32, 2), sizes[0].descriptor_count);
    try testing.expectEqual(vk.DescriptorType.storage_buffer_dynamic, sizes[1].type);
    try testing.expectEqual(@as(u32, 1), sizes[1].descriptor_count);
    try testing.expectEqual(vk.DescriptorType.storage_buffer, sizes[2].type);
    try testing.expectEqual(@as(u32, 1), sizes[2].descriptor_count);
}

test "an array binding contributes its whole length" {
    // The material set: five textures behind one slot, which is one binding and
    // five descriptors. Counting bindings here would undersize the pool by four.
    const material_set = [_]gpu.DescriptorBinding{
        .{ .slot = 0, .kind = .combined_image_sampler, .stages = fragment, .count = 5 },
        .{ .slot = 1, .kind = .storage_buffer, .stages = vertex },
    };
    const sizes = gpu.DescriptorSets(&material_set).pool_sizes_per_set;

    try testing.expectEqual(@as(usize, 2), sizes.len);
    try testing.expectEqual(@as(u32, 5), sizes[0].descriptor_count);
    try testing.expectEqual(@as(u32, 1), sizes[1].descriptor_count);
}

test "the layout mirrors the binding list entry for entry" {
    // An array binding is in the list on purpose: a layout that reports one
    // descriptor where the pool was sized for five is a mismatch the pool sizes
    // alone cannot show.
    const mixed = frame_set ++ [_]gpu.DescriptorBinding{
        .{ .slot = 4, .kind = .combined_image_sampler, .stages = fragment, .count = 5 },
    };
    const entries = gpu.DescriptorSets(&mixed).layout_bindings;

    try testing.expectEqual(mixed.len, entries.len);
    for (mixed, entries) |binding, entry| {
        try testing.expectEqual(binding.slot, entry.binding);
        try testing.expectEqual(binding.kind, entry.descriptor_type);
        try testing.expectEqual(binding.count, entry.descriptor_count);
        try testing.expectEqual(binding.stages, entry.stage_flags);
    }
    try testing.expectEqual(@as(?[*]const vk.Sampler, null), entries[0].p_immutable_samplers);
}

test "a set of one binding is the shape every backend pass wants" {
    // The sequence this type exists to remove: one uniform binding, sized for
    // as many sets as the caller asks for.
    const single = [_]gpu.DescriptorBinding{
        .{ .slot = 0, .kind = .uniform_buffer, .stages = fragment },
    };
    const sizes = gpu.DescriptorSets(&single).pool_sizes_per_set;

    try testing.expectEqual(@as(usize, 1), sizes.len);
    try testing.expectEqual(@as(u32, 1), sizes[0].descriptor_count);
}

test "the group's own methods are reached by the compiler" {
    // Each one needs a device to call, and a test reaches only what it calls.
    const Group = gpu.DescriptorSets(&frame_set);
    _ = &Group.init;
    _ = &Group.deinit;
    _ = &Group.set;
}
