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
    .image = @fromBackingInt(@intCast(7)),
    .view = @fromBackingInt(@intCast(8)),
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

test "the transition chains after the acquire wait, and presentation waits on the signal" {
    // The acquire semaphore's wait reaches only the stage the submission names,
    // so the begin barrier's source scope names it too, with no access: the
    // semaphore made the image available. The signalled semaphore orders
    // presentation after the pass, so the end barrier's destination is empty.
    const begin = post.beginBarriers(target);
    const end = post.endBarriers(target);

    try testing.expectEqual(vk.PipelineStageFlags2{ .color_attachment_output_bit = true }, begin[0].src_stage_mask);
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
    // Slot zero is the HDR target, and the layout its descriptor declares is the
    // one `pass.end` transitions that target into. A different layout is a
    // validation error at draw time and nothing sooner.
    try testing.expectEqual(@as(u32, 0), gpu.PostBindings[0].slot);
    try testing.expectEqual(vk.DescriptorType.combined_image_sampler, gpu.PostBindings[0].kind);
    try testing.expect(gpu.PostBindings[0].stages.fragment_bit);

    const main_end = gpu.MainPass.endBarriers(.{
        .hdr_image = @fromBackingInt(@intCast(1)),
        .hdr_view = @fromBackingInt(@intCast(2)),
        .depth_image = @fromBackingInt(@intCast(3)),
        .depth_view = @fromBackingInt(@intCast(4)),
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

test "the engine's two words lead the push block, and the application's part fills it to 128 bytes" {
    try testing.expectEqual(@as(usize, 128), @sizeOf(post.PushConstants));
    try testing.expectEqual(@as(usize, 4), @alignOf(post.PushConstants));
    try testing.expectEqual(@as(usize, 0), @offsetOf(post.PushConstants, "exposure"));
    try testing.expectEqual(@as(usize, 4), @offsetOf(post.PushConstants, "bloom"));
    try testing.expectEqual(@as(usize, 16), @offsetOf(post.PushConstants, "application"));

    try testing.expectEqual(@as(u32, 0), post.push_constant_range.offset);
    try testing.expectEqual(@as(u32, @sizeOf(post.PushConstants)), post.push_constant_range.size);
    try testing.expectEqual(
        vk.ShaderStageFlags{ .fragment_bit = true },
        post.push_constant_range.stage_flags,
    );

    const constants = try post.pushConstants(.{}, null);
    try testing.expectEqual(@as(f32, 1), constants.exposure);
    try testing.expectEqual(post.Application.none.bytes, constants.application);
}

test "an application's block reaches the push constants byte for byte, the rest zero" {
    const Fog = extern struct {
        colour: [4]f32,
        start: f32,
        end: f32,
    };
    const fog: Fog = .{ .colour = .{ 0.2, 0.03, 0.03, 1 }, .start = 10, .end = 96 };
    const constants = try post.pushConstants(.{ .application = .of(fog) }, null);

    try testing.expectEqualSlices(u8, std.mem.asBytes(&fog), constants.application[0..@sizeOf(Fog)]);
    for (constants.application[@sizeOf(Fog)..]) |byte| try testing.expectEqual(@as(u8, 0), byte);
}

test "only finite non-negative exposure reaches command state" {
    try testing.expectError(error.InvalidExposure, post.pushConstants(.{ .exposure = -1 }, null));
    try testing.expectError(error.InvalidExposure, post.pushConstants(.{ .exposure = std.math.nan(f32) }, null));
    try testing.expectError(error.InvalidExposure, post.pushConstants(.{ .exposure = std.math.inf(f32) }, null));
    try testing.expectError(error.InvalidExposure, post.pushConstants(.{ .exposure = -std.math.inf(f32) }, null));
    try testing.expectError(error.InvalidExposure, post.exposure(.{ .exposure = -1 }));

    const zero = try post.pushConstants(.{ .exposure = 0 }, null);
    try testing.expectEqual(@as(f32, 0), zero.exposure);
}

test "a recording with no chain composites nothing, and says so with a weight" {
    // The weight is zero without a look, but that is not what switches the
    // composite off: the pipeline built on the entry point that never samples
    // the chain is. A chain nothing recorded this frame holds whatever its
    // memory held, and an unsigned float format has bit patterns that decode to
    // NaN, which multiplying by zero does not remove.
    const without = try post.pushConstants(.{}, null);
    try testing.expectEqual(@as(f32, 0), without.bloom);

    const look = try gpu.bloomResolve(.{}, 7);
    const with = try post.pushConstants(.{}, look);
    try testing.expectEqual(look.composite, with.bloom);
    try testing.expect(with.bloom > 0);
}

test "the post set names the target and the chain sampled, then depth read by texel" {
    try testing.expectEqual(@as(usize, 3), post.bindings.len);
    for (post.bindings, 0..) |binding, slot| {
        try testing.expectEqual(@as(u32, @intCast(slot)), binding.slot);
        try testing.expect(binding.stages.fragment_bit);
    }
    try testing.expectEqual(vk.DescriptorType.combined_image_sampler, post.bindings[0].kind);
    try testing.expectEqual(vk.DescriptorType.combined_image_sampler, post.bindings[1].kind);
    try testing.expectEqual(vk.DescriptorType.sampled_image, post.bindings[2].kind);
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
