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

// Material 0 is solid, material 1 has never been pointed at its textures, and
// material 2 blends.
const modes = [_]?gpu.PipelineMode{ .solid, null, .blended };

test "a record batch names configured state and a live instance range" {
    // Exact fit at the end of the uploaded instance slice.
    try gpu.validateRendererRecordBatch(&modes, 5, 0, .{ .back_bit = true }, 3, 2, false);
    try gpu.validateRendererRecordBatch(&modes, 5, 0, .{}, 0, 5, false);

    try testing.expectError(
        error.MaterialIndexOutOfRange,
        gpu.validateRendererRecordBatch(&modes, 5, 3, .{}, 0, 1, false),
    );
    try testing.expectError(
        error.MaterialNotConfigured,
        gpu.validateRendererRecordBatch(&modes, 5, 1, .{}, 0, 1, false),
    );
    try testing.expectError(
        error.EmptyBatch,
        gpu.validateRendererRecordBatch(&modes, 5, 0, .{}, 0, 0, false),
    );
    try testing.expectError(
        error.UnsupportedCullMode,
        gpu.validateRendererRecordBatch(
            &modes,
            5,
            0,
            .{ .front_bit = true, .back_bit = true },
            0,
            1,
            false,
        ),
    );
    try testing.expectError(
        error.InstanceRangeOutOfBounds,
        gpu.validateRendererRecordBatch(&modes, 5, 0, .{}, 4, 2, false),
    );
    // Subtraction after testing `first` avoids overflow even for an arbitrary
    // boundary value supplied by composition.
    try testing.expectError(
        error.InstanceRangeOutOfBounds,
        gpu.validateRendererRecordBatch(&modes, 5, 0, .{}, std.math.maxInt(u32), 1, false),
    );
}

test "a solid batch is refused once the list has reached its blended run" {
    // What the ordering buys is that a blended surface is composited over the
    // scene behind it. A solid batch recorded afterwards would be composited
    // over the blended one instead, and its depth write would then reject the
    // blended fragments already in the attachment.
    try testing.expectError(
        error.SolidBatchAfterBlended,
        gpu.validateRendererRecordBatch(&modes, 5, 0, .{}, 0, 1, true),
    );

    // A blended batch after a blended one is the ordinary case.
    try gpu.validateRendererRecordBatch(&modes, 5, 2, .{}, 0, 1, true);
    // And a blended batch may open the run.
    try gpu.validateRendererRecordBatch(&modes, 5, 2, .{}, 0, 1, false);
}

test "the layer order is checked against the material, not the batch's position" {
    // An unconfigured material is refused whichever layer the list has reached,
    // so the ordering check cannot mask a missing descriptor set.
    try testing.expectError(
        error.MaterialNotConfigured,
        gpu.validateRendererRecordBatch(&modes, 5, 1, .{}, 0, 1, true),
    );
    // And an index past the table is still out of range, which has to be caught
    // before the mode is read out of it.
    try testing.expectError(
        error.MaterialIndexOutOfRange,
        gpu.validateRendererRecordBatch(&modes, 5, 3, .{}, 0, 1, true),
    );
}

test "the three glTF alpha modes collapse onto the two pipelines" {
    // MASK has no pipeline of its own: a masked fragment either survives the
    // cutoff and behaves exactly like an opaque one or is discarded.
    try testing.expectEqual(gpu.PipelineMode.solid, gpu.pipelineModeFor(.@"opaque"));
    try testing.expectEqual(gpu.PipelineMode.solid, gpu.pipelineModeFor(.mask));
    try testing.expectEqual(gpu.PipelineMode.blended, gpu.pipelineModeFor(.blend));
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
