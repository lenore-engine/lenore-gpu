const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;
const post = gpu.PostPass;

fn expectColour(expected: [3]f32, actual: [3]f32) !void {
    inline for (0..3) |channel|
        try testing.expectApproxEqAbs(expected[channel], actual[channel], 1.0e-6);
}

const target: gpu.PostTarget = .{
    .image = @enumFromInt(7),
    .view = @enumFromInt(8),
    .extent = .{ .width = 800, .height = 600 },
};

test "the acquired image is left ready to present" {
    const begin = post.beginBarriers(target);
    const end = post.endBarriers(target);

    try testing.expectEqual(target.image, begin[0].image);
    try testing.expectEqual(target.image, end[0].image);

    // The pair has to meet in the middle, or the pass renders into one layout
    // and hands over another.
    try testing.expectEqual(begin[0].new_layout, end[0].old_layout);
    try testing.expectEqual(vk.ImageLayout.present_src_khr, end[0].new_layout);
}

test "the semaphores carry the scopes the barriers leave empty" {
    // The acquire semaphore orders this frame's writes after the previous
    // presentation, and the signalled semaphore orders presentation after
    // them. Naming a stage on those sides as well would claim a dependency
    // that the submission already provides.
    const begin = post.beginBarriers(target);
    const end = post.endBarriers(target);

    try testing.expectEqual(vk.PipelineStageFlags2{}, begin[0].src_stage_mask);
    try testing.expectEqual(vk.AccessFlags2{}, begin[0].src_access_mask);
    try testing.expectEqual(vk.PipelineStageFlags2{}, end[0].dst_stage_mask);
    try testing.expectEqual(vk.AccessFlags2{}, end[0].dst_access_mask);

    // The two scopes that do matter: the pass writes as a colour attachment.
    try testing.expect(begin[0].dst_stage_mask.color_attachment_output_bit);
    try testing.expect(end[0].src_stage_mask.color_attachment_output_bit);
}

test "the presentable image is written whole and never read first" {
    const colour = post.colourAttachment(target);

    // Every pixel is covered by the triangle, so loading the previous contents
    // is bandwidth spent on values that are all replaced.
    try testing.expectEqual(vk.AttachmentLoadOp.dont_care, colour.load_op);
    try testing.expectEqual(vk.AttachmentStoreOp.store, colour.store_op);
    try testing.expectEqual(target.view, colour.image_view);
    try testing.expectEqual(post.beginBarriers(target)[0].new_layout, colour.image_layout);
}

test "the pass samples what the main pass left behind" {
    // One binding, and its layout is the one `pass.end` transitions the HDR
    // target into. A different layout here is a validation error at draw time
    // and nothing sooner.
    try testing.expectEqual(@as(usize, 1), gpu.PostBindings.len);
    try testing.expectEqual(vk.DescriptorType.combined_image_sampler, gpu.PostBindings[0].kind);
    try testing.expect(gpu.PostBindings[0].stages.fragment_bit);

    const main_end = gpu.MainPass.endBarriers(.{
        .hdr_image = @enumFromInt(1),
        .hdr_view = @enumFromInt(2),
        .depth_image = @enumFromInt(3),
        .depth_view = @enumFromInt(4),
        .extent = target.extent,
    });
    try testing.expectEqual(vk.ImageLayout.shader_read_only_optimal, main_end[0].new_layout);
}

test "three vertices cover the screen with no buffer bound" {
    try testing.expectEqual(@as(u32, 3), post.vertex_count);
}

test "the post entry points are reached by the compiler" {
    _ = &post.begin;
    _ = &post.end;
    _ = &post.write;
}

test "the runtime look fits one fragment push-constant word" {
    try testing.expectEqual(@as(usize, 4), @sizeOf(post.PushConstants));
    try testing.expectEqual(@as(usize, 4), @alignOf(post.PushConstants));
    try testing.expectEqual(@as(usize, 0), @offsetOf(post.PushConstants, "exposure"));

    try testing.expectEqual(@as(u32, 0), post.push_constant_range.offset);
    try testing.expectEqual(@as(u32, @sizeOf(post.PushConstants)), post.push_constant_range.size);
    try testing.expectEqual(
        vk.ShaderStageFlags{ .fragment_bit = true },
        post.push_constant_range.stage_flags,
    );

    const constants = try post.pushConstants(.{});
    try testing.expectEqual(@as(f32, 1), constants.exposure);
}

test "only finite non-negative exposure reaches command state" {
    try testing.expectError(error.InvalidExposure, post.pushConstants(.{ .exposure = -1 }));
    try testing.expectError(error.InvalidExposure, post.pushConstants(.{ .exposure = std.math.nan(f32) }));
    try testing.expectError(error.InvalidExposure, post.pushConstants(.{ .exposure = std.math.inf(f32) }));
    try testing.expectError(error.InvalidExposure, post.pushConstants(.{ .exposure = -std.math.inf(f32) }));

    const zero = try post.pushConstants(.{ .exposure = 0 });
    try testing.expectEqual(@as(f32, 0), zero.exposure);
}

test "PBR Neutral preserves its near-black and uncompressed branches" {
    try expectColour(
        .{ 0.01, 0.07, 0.17 },
        try post.toneMap(.{ 0.04, 0.10, 0.20 }, .{}),
    );
    try expectColour(
        .{ 0.16, 0.36, 0.56 },
        try post.toneMap(.{ 0.20, 0.40, 0.60 }, .{}),
    );
}

test "PBR Neutral compresses and desaturates an HDR highlight" {
    const mapped = try post.toneMap(.{ 4, 1, 0.25 }, .{});
    try expectColour(.{ 0.9832558, 0.4682992, 0.3395600 }, mapped);
    for (mapped) |channel| try testing.expect(channel >= 0 and channel <= 1);
}

test "exposure scales linear radiance before the operator" {
    try expectColour(
        .{ 0.3050976, 0.6075488, 0.91 },
        try post.toneMap(.{ 0.20, 0.40, 0.60 }, .{ .exposure = 2 }),
    );
}

test "negative radiance is removed before the operator" {
    try expectColour(
        .{ 0, 0.10, 0.20 },
        try post.toneMap(.{ -1, 0.10, 0.20 }, .{}),
    );
}
