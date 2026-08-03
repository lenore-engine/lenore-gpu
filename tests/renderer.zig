const std = @import("std");
const res = @import("lenore-resources");
const gpu = @import("lenore-gpu");

const testing = std.testing;

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
