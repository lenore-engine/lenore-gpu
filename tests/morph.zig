const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;

test "a mesh is covered by whole groups, rounded up" {
    const size = gpu.morphGroupSize;

    // An exact multiple takes no extra group, and one vertex past it takes one.
    // Rounding down instead leaves the tail of the mesh at its bind shape, which
    // reads as a partly animated model rather than as an error.
    try testing.expectEqual(@as(u32, 1), gpu.morphGroupsFor(size));
    try testing.expectEqual(@as(u32, 2), gpu.morphGroupsFor(size + 1));
    try testing.expectEqual(@as(u32, 2), gpu.morphGroupsFor(size * 2));
    try testing.expectEqual(@as(u32, 3), gpu.morphGroupsFor(size * 2 + 1));

    // One vertex is one group, not none.
    try testing.expectEqual(@as(u32, 1), gpu.morphGroupsFor(1));

    // The property both cases above are instances of: every vertex is covered,
    // and never by more than one group's worth of slack.
    for ([_]u32{ 1, 2, 63, 64, 65, 1504, 944_100 }) |count| {
        const groups = gpu.morphGroupsFor(count);
        const covered = groups * size;
        try testing.expect(covered >= count);
        try testing.expect(covered - count < size);
    }
}

test "a destination too large for its own index is refused" {
    try testing.expectEqual(@as(u32, 6), try gpu.morphDestinationElements(2, 3));

    // The mesh is bounded by a u32 vertex count and the destination holds one
    // copy per frame in flight, so the product is what can leave the range the
    // push constant addressing it has.
    const highest = std.math.maxInt(u32);
    try testing.expectEqual(highest, try gpu.morphDestinationElements(1, highest));
    try testing.expectError(
        error.DestinationTooLarge,
        gpu.morphDestinationElements(2, highest),
    );
    try testing.expectError(
        error.DestinationTooLarge,
        gpu.morphDestinationElements(std.math.maxInt(usize), 2),
    );
}

test "the prepass barrier releases to the vertex fetch, not to a shader" {
    const released = gpu.morphDependency();

    try testing.expect(released.src_stage.compute_shader_bit);
    try testing.expect(released.src_access.shader_storage_write_bit);

    // The destination is read by the vertex input stage, which runs before the
    // vertex shader. Naming `vertex_shader_bit` here orders nothing that reads
    // this buffer, and no layer reports it: the barrier is legal and the fetch
    // is simply unsynchronised.
    try testing.expect(released.dst_stage.vertex_attribute_input_bit);
    try testing.expect(!released.dst_stage.vertex_shader_bit);
    try testing.expect(released.dst_access.vertex_attribute_read_bit);
}

test "exactly one binding of the prepass set is dynamic" {
    // `record` derives its dynamic offset array from this table and passes one
    // value. Vulkan consumes one offset per dynamic descriptor in increasing
    // binding order, so a second dynamic binding here has to break the build
    // rather than hand the weights the other one's offset.
    var dynamic: usize = 0;
    var dynamic_slot: u32 = 0;
    for (gpu.morph_bindings) |binding| {
        try testing.expect(binding.stages.compute_bit);
        switch (binding.kind) {
            .storage_buffer_dynamic, .uniform_buffer_dynamic => {
                dynamic += 1;
                dynamic_slot = binding.slot;
            },
            .storage_buffer => {},
            else => return error.TestUnexpectedDescriptorKind,
        }
    }
    try testing.expectEqual(@as(usize, 1), dynamic);
    try testing.expectEqual(@as(u32, 2), dynamic_slot);
}

test "the push block is the size the range declares" {
    // A range shorter than the block leaves the tail unwritten, and the shader
    // reads it as whatever the last pipeline pushed.
    try testing.expectEqual(
        @as(u32, @sizeOf(gpu.MorphPushConstants)),
        gpu.morphPushConstantRange.size,
    );
    try testing.expectEqual(@as(u32, 0), gpu.morphPushConstantRange.offset);
    try testing.expect(gpu.morphPushConstantRange.stage_flags.compute_bit);
    try testing.expect(!gpu.morphPushConstantRange.stage_flags.vertex_bit);
}

test "weights are reserved end to end and the bound is the slot" {
    // Three meshes of two, one and three targets laid out in registration order.
    // Overlapping runs would have one mesh's blend drive another's shape, which
    // no layer reports and which a two-mesh model with equal counts hides.
    var used: usize = 0;
    var bases: [3]u32 = undefined;
    for ([_]u32{ 2, 1, 3 }, &bases) |targets, *base| {
        const reserved = try gpu.morphReserveWeights(6, used, targets);
        base.* = reserved.base;
        used = reserved.used;
    }
    try testing.expectEqualSlices(u32, &.{ 0, 2, 3 }, &bases);
    try testing.expectEqual(@as(usize, 6), used);

    // Exactly full is not an error, and one past it is.
    try testing.expectError(
        error.WeightCapacityExceeded,
        gpu.morphReserveWeights(6, used, 1),
    );
    try testing.expectError(
        error.WeightCapacityExceeded,
        gpu.morphReserveWeights(4, 2, 3),
    );
    // A pass built with no weight capacity registers nothing rather than
    // reserving from an empty slot.
    try testing.expectError(
        error.WeightCapacityExceeded,
        gpu.morphReserveWeights(0, 0, 1),
    );
    try testing.expectError(
        error.WeightCapacityExceeded,
        gpu.morphReserveWeights(2, 2, std.math.maxInt(u32)),
    );
}

test "a registration answers to the descriptor set count" {
    try gpu.morphValidateRegistration(0, 1);
    try gpu.morphValidateRegistration(3, 4);
    try testing.expectError(error.MeshCapacityExceeded, gpu.morphValidateRegistration(4, 4));
    try testing.expectError(error.MeshCapacityExceeded, gpu.morphValidateRegistration(0, 0));
}

test "a weight array of the wrong length is refused" {
    try gpu.morphValidateWeightCount(2, 2);
    // Both directions: too few leaves the tail as another mesh wrote it, and too
    // many is a caller that thinks this mesh has targets it does not.
    try testing.expectError(error.WeightCountMismatch, gpu.morphValidateWeightCount(2, 1));
    try testing.expectError(error.WeightCountMismatch, gpu.morphValidateWeightCount(2, 3));
    try testing.expectError(error.WeightCountMismatch, gpu.morphValidateWeightCount(1, 0));
}

test "a batch with no prepass output fetches the mesh's own vertices" {
    // `batchVertexSource` is what the recorder caches its vertex binding on, so
    // the resolution has to be the same function the bind uses. A batch that
    // resolved to the mesh here and bound something else would rebind on every
    // draw, which is slower and still correct; the reverse silently draws one
    // registration's shape for another's.
    var mesh: gpu.Mesh = undefined;
    mesh.vertex_buffer.handle = @enumFromInt(0xf00d);

    const own = gpu.batchVertexSource(.{
        .mesh = &mesh,
        .material_index = 0,
        .cull_mode = .{},
        .front_face = .counter_clockwise,
        .first_instance = 0,
        .instance_count = 1,
    });
    try testing.expectEqual(mesh.vertex_buffer.handle, own.handle);
    try testing.expectEqual(@as(u64, 0), own.offset);

    const substituted = gpu.batchVertexSource(.{
        .mesh = &mesh,
        .material_index = 0,
        .cull_mode = .{},
        .front_face = .counter_clockwise,
        .first_instance = 0,
        .instance_count = 1,
        .vertex_source = .{ .handle = @enumFromInt(0xbeef), .offset = 48 },
    });
    try testing.expectEqual(@as(vk.Buffer, @enumFromInt(0xbeef)), substituted.handle);
    try testing.expectEqual(@as(u64, 48), substituted.offset);
}

test "only a morphed mesh's vertex buffer is also a storage buffer" {
    // The prepass reads the base vertices through a storage descriptor, so the
    // buffer has to declare that usage. Nothing offline sees the omission: it
    // surfaces as the layer refusing the descriptor write at registration, on a
    // model that has morph targets, which is the one case a static scene never
    // reaches.
    const plain = gpu.meshVertexUsage(0);
    try testing.expect(plain.vertex_buffer_bit);
    try testing.expect(!plain.storage_buffer_bit);

    const morphed = gpu.meshVertexUsage(2);
    try testing.expect(morphed.vertex_buffer_bit);
    try testing.expect(morphed.storage_buffer_bit);
}
