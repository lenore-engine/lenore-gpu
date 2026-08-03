const std = @import("std");
const res = @import("lenore-resources");
const gpu = @import("lenore-gpu");

const testing = std.testing;

test "a material index addresses one of the allocated material sets" {
    try gpu.validateRendererMaterialIndex(2, 0);
    try gpu.validateRendererMaterialIndex(2, 1);

    // Count is an exclusive bound. The first index past it is the boundary an
    // off-by-one would admit to `DescriptorSets.set`, where ReleaseFast has no
    // assertion left to stop the out-of-bounds read.
    try testing.expectError(
        error.MaterialIndexOutOfRange,
        gpu.validateRendererMaterialIndex(2, 2),
    );
    try testing.expectError(
        error.MaterialIndexOutOfRange,
        gpu.validateRendererMaterialIndex(0, 0),
    );
}

test "a renderer frame index addresses a tracked frame slot" {
    try gpu.validateRendererFrameIndex(2, 0);
    try gpu.validateRendererFrameIndex(2, 1);
    try testing.expectError(error.FrameIndexOutOfRange, gpu.validateRendererFrameIndex(2, 2));
    try testing.expectError(error.FrameIndexOutOfRange, gpu.validateRendererFrameIndex(0, 0));
}

test "a record batch names configured state and a live instance range" {
    const ready = [_]bool{ true, false };

    // Exact fit at the end of the uploaded instance slice.
    try gpu.validateRendererRecordBatch(&ready, 5, 0, .{ .back_bit = true }, 3, 2);
    try gpu.validateRendererRecordBatch(&ready, 5, 0, .{}, 0, 5);

    try testing.expectError(
        error.MaterialIndexOutOfRange,
        gpu.validateRendererRecordBatch(&ready, 5, 2, .{}, 0, 1),
    );
    try testing.expectError(
        error.MaterialNotConfigured,
        gpu.validateRendererRecordBatch(&ready, 5, 1, .{}, 0, 1),
    );
    try testing.expectError(
        error.EmptyBatch,
        gpu.validateRendererRecordBatch(&ready, 5, 0, .{}, 0, 0),
    );
    try testing.expectError(
        error.UnsupportedCullMode,
        gpu.validateRendererRecordBatch(&ready, 5, 0, .{ .front_bit = true, .back_bit = true }, 0, 1),
    );
    try testing.expectError(
        error.InstanceRangeOutOfBounds,
        gpu.validateRendererRecordBatch(&ready, 5, 0, .{}, 4, 2),
    );
    // Subtraction after testing `first` avoids overflow even for an arbitrary
    // boundary value supplied by composition.
    try testing.expectError(
        error.InstanceRangeOutOfBounds,
        gpu.validateRendererRecordBatch(&ready, 5, 0, .{}, std.math.maxInt(u32), 1),
    );
}

test "a mesh draws through the pipeline its own streams call for" {
    // The skinned pipeline is the only one that declares binding 1 and the only
    // entry point that reads the joint array, so this choice is what stands
    // between a skinned mesh and its bind pose.
    try testing.expectEqual(gpu.SceneVariant.unskinned, gpu.sceneVariantFor(.{}));
    try testing.expectEqual(gpu.SceneVariant.skinned, gpu.sceneVariantFor(.{ .skinned = true }));
}

test "the streams with no shader path do not select a variant of their own" {
    // Colour and the second UV set are carried by a mesh and bound by
    // `Mesh.bind`, and no entry point reads either. A mesh with them draws
    // through the same pipeline as a mesh without.
    try testing.expectEqual(
        gpu.SceneVariant.unskinned,
        gpu.sceneVariantFor(.{ .colour = true, .uv1 = true }),
    );

    // But they must not mask skinning: every combination that carries the skin
    // stream selects the pipeline that declares it.
    for (0..8) |bits| {
        const streams: res.VertexStreams = @bitCast(@as(u3, @intCast(bits)));
        const expected: gpu.SceneVariant = if (streams.skinned) .skinned else .unskinned;
        try testing.expectEqual(expected, gpu.sceneVariantFor(streams));
    }
}
