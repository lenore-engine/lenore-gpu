const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;

const vertex: vk.ShaderStageFlags = .{ .vertex_bit = true };
const fragment: vk.ShaderStageFlags = .{ .fragment_bit = true };

// The set the first lit frame needs, as the shader will declare it: the
// per-frame buffers carry a dynamic offset so that one set serves every frame.
const frame_set = [_]gpu.DescriptorBinding{
    .{ .slot = 0, .name = "camera", .kind = .uniform_buffer_dynamic, .stages = vertex },
    .{ .slot = 1, .name = "instances", .kind = .storage_buffer_dynamic, .stages = vertex },
    .{ .slot = 2, .name = "lights", .kind = .uniform_buffer_dynamic, .stages = fragment },
    .{ .slot = 3, .name = "materials", .kind = .storage_buffer, .stages = fragment },
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
        .{ .slot = 0, .name = "textures", .kind = .combined_image_sampler, .stages = fragment, .count = 5 },
        .{ .slot = 1, .name = "extra", .kind = .storage_buffer, .stages = vertex },
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
        .{ .slot = 4, .name = "atlas", .kind = .combined_image_sampler, .stages = fragment, .count = 5 },
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
        .{ .slot = 0, .name = "block", .kind = .uniform_buffer, .stages = fragment },
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
    _ = &Group.writeBuffers;
    _ = &gpu.DescriptorSets(&sampled_set).writeImages;
}

// A buffer whose only read field is its handle. Building the write is arithmetic
// over the binding list and an opaque handle, so it needs no device and no
// allocation; every other field of a `Buffer` belongs to the allocation that a
// real one carries.
fn handled(id: u64) gpu.Buffer {
    var buffer: gpu.Buffer = undefined;
    buffer.handle = @fromBackingInt(@intCast(id));
    return buffer;
}

const sampled_set = [_]gpu.DescriptorBinding{
    .{ .slot = 0, .name = "first", .kind = .combined_image_sampler, .stages = fragment },
    .{ .slot = 1, .name = "second", .kind = .combined_image_sampler, .stages = fragment },
};

const mixed_acceleration_set = [_]gpu.DescriptorBinding{
    .{ .slot = 0, .name = "data", .kind = .storage_buffer, .stages = .{ .compute_bit = true } },
    .{ .slot = 1, .name = "scene", .kind = .acceleration_structure_khr, .stages = .{ .compute_bit = true } },
};

test "a buffer write is keyed by binding name, not by position" {
    const Group = gpu.DescriptorSets(&frame_set);
    var first = handled(0x11);
    var second = handled(0x22);
    var third = handled(0x33);
    var fourth = handled(0x44);

    var infos: [frame_set.len]vk.DescriptorBufferInfo = undefined;
    // Both spellings in one list, which is the case the resolution exists for: a
    // bare buffer is the whole of it, and a source that is not says so.
    const Source = gpu.DescriptorBufferSource;
    // Deliberately not in the layout's order, which is the property: the name
    // chooses the slot, so a list written in any order lands the same way and
    // two entries exchanged is not a mistake that can be made.
    const writes = Group.bufferWrites(.null_handle, &infos, .{
        .materials = &fourth,
        .camera = &first,
        .lights = Source{ .buffer = &third, .offset = 64, .range = 128 },
        .instances = Source{ .buffer = &second, .offset = 256 },
    });

    // The slot and the type come from the binding, never from the source.
    for (frame_set, writes) |binding, write| {
        try testing.expectEqual(binding.slot, write.dst_binding);
        try testing.expectEqual(binding.kind, write.descriptor_type);
        try testing.expectEqual(@as(u32, 1), write.descriptor_count);
        try testing.expectEqual(@as(u32, 0), write.dst_array_element);
        try testing.expectEqual(vk.DescriptorSet.null_handle, write.dst_set);
    }

    try testing.expectEqual(@as(vk.Buffer, @fromBackingInt(@intCast(0x11))), infos[0].buffer);
    try testing.expectEqual(@as(vk.Buffer, @fromBackingInt(@intCast(0x44))), infos[3].buffer);

    // An absent range is the rest of the buffer, and an offset given without one
    // does not silently become a length.
    try testing.expectEqual(vk.WHOLE_SIZE, infos[0].range);
    try testing.expectEqual(@as(vk.DeviceSize, 0), infos[0].offset);
    try testing.expectEqual(@as(vk.DeviceSize, 256), infos[1].offset);
    try testing.expectEqual(vk.WHOLE_SIZE, infos[1].range);
    try testing.expectEqual(@as(vk.DeviceSize, 128), infos[2].range);

    // Each write points at its own info. One shared info would hand every
    // binding the last buffer written, which is a set that is internally
    // consistent and completely wrong.
    for (writes, &infos) |write, *info| {
        const expected: @TypeOf(write.p_buffer_info) = @ptrCast(info);
        try testing.expectEqual(expected, write.p_buffer_info);
    }
}

test "one buffer can fill two slots" {
    // What a shader needs when the same storage buffer is declared read-write
    // for one stage and read-only for another. The pair is two bindings and two
    // writes, so the mapping is per binding rather than per buffer.
    const shared = [_]gpu.DescriptorBinding{
        .{ .slot = 0, .name = "shared_rw", .kind = .storage_buffer, .stages = .{ .compute_bit = true } },
        .{ .slot = 1, .name = "shared_read", .kind = .storage_buffer, .stages = fragment },
    };
    var buffer = handled(0x99);

    var infos: [shared.len]vk.DescriptorBufferInfo = undefined;
    const writes = gpu.DescriptorSets(&shared).bufferWrites(.null_handle, &infos, .{
        .shared_rw = &buffer,
        .shared_read = &buffer,
    });

    try testing.expectEqual(@as(usize, 2), writes.len);
    try testing.expectEqual(infos[0].buffer, infos[1].buffer);
    try testing.expectEqual(@as(u32, 0), writes[0].dst_binding);
    try testing.expectEqual(@as(u32, 1), writes[1].dst_binding);
}

test "mixed buffer and acceleration writes select their own bindings" {
    const Group = gpu.DescriptorSets(&mixed_acceleration_set);
    var data = handled(0x55);
    var buffer_infos: [1]vk.DescriptorBufferInfo = undefined;
    const buffer_writes = Group.bufferWrites(.null_handle, &buffer_infos, .{ .data = &data });

    try testing.expectEqual(@as(usize, 1), buffer_writes.len);
    try testing.expectEqual(@as(u32, 0), buffer_writes[0].dst_binding);
    try testing.expectEqual(@as(vk.Buffer, @fromBackingInt(@intCast(0x55))), buffer_infos[0].buffer);

    var structure: gpu.AccelerationStructure = undefined;
    structure.handle = @fromBackingInt(@intCast(0x66));
    const source = gpu.DescriptorAccelerationSource.one(&structure);
    var acceleration_infos: [1]vk.WriteDescriptorSetAccelerationStructureKHR = undefined;
    const acceleration_writes = try Group.accelerationWrites(
        .null_handle,
        &acceleration_infos,
        .{ .scene = source },
    );

    try testing.expectEqual(@as(usize, 1), acceleration_writes.len);
    try testing.expectEqual(@as(u32, 1), acceleration_writes[0].dst_binding);
    try testing.expectEqual(@as(u32, 1), acceleration_writes[0].descriptor_count);
    try testing.expectEqual(@as(u32, 1), acceleration_infos[0].acceleration_structure_count);
    try testing.expectEqual(
        @as(?*const anyopaque, @ptrCast(&acceleration_infos[0])),
        acceleration_writes[0].p_next,
    );
    try testing.expectEqual(structure.handle, acceleration_infos[0].p_acceleration_structures[0]);
}

test "an acceleration write rejects a source count that differs from its binding" {
    const Group = gpu.DescriptorSets(&mixed_acceleration_set);
    const no_handles = [_]vk.AccelerationStructureKHR{};
    var infos: [1]vk.WriteDescriptorSetAccelerationStructureKHR = undefined;

    try testing.expectError(error.DescriptorCountMismatch, Group.accelerationWrites(
        .null_handle,
        &infos,
        .{ .scene = gpu.DescriptorAccelerationSource{ .handles = &no_handles } },
    ));
}

test "an image write carries the layout the caller states" {
    const Group = gpu.DescriptorSets(&sampled_set);
    var infos: [sampled_set.len]vk.DescriptorImageInfo = undefined;
    const Source = gpu.DescriptorImageSource;
    const writes = Group.imageWrites(.null_handle, &infos, .{
        .first = Source{
            .view = @fromBackingInt(@intCast(0xa1)),
            .sampler = @fromBackingInt(@intCast(0xb1)),
            .layout = .shader_read_only_optimal,
        },
        .second = Source{
            .view = @fromBackingInt(@intCast(0xa2)),
            .sampler = @fromBackingInt(@intCast(0xb2)),
            .layout = .general,
        },
    });

    // Not defaulted and not inferred: the layout an image is in is a fact about
    // the transition that put it there, and a descriptor naming another one is a
    // validation error at draw time and nothing sooner.
    try testing.expectEqual(vk.ImageLayout.shader_read_only_optimal, infos[0].image_layout);
    try testing.expectEqual(vk.ImageLayout.general, infos[1].image_layout);

    try testing.expectEqual(@as(vk.ImageView, @fromBackingInt(@intCast(0xa1))), infos[0].image_view);
    try testing.expectEqual(@as(vk.Sampler, @fromBackingInt(@intCast(0xb2))), infos[1].sampler);

    for (sampled_set, writes, &infos) |binding, write, *info| {
        try testing.expectEqual(binding.slot, write.dst_binding);
        try testing.expectEqual(vk.DescriptorType.combined_image_sampler, write.descriptor_type);
        const expected: @TypeOf(write.p_image_info) = @ptrCast(info);
        try testing.expectEqual(expected, write.p_image_info);
    }
}
