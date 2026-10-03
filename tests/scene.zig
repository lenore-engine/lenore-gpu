const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;
const pass = gpu.MainPass;

// Handles are only compared here, never dereferenced, so distinct synthetic
// ones are enough to tell the two attachments apart.
const target: gpu.MainPassTarget = .{
    .hdr_image = @fromBackingInt(@intCast(1)),
    .hdr_view = @fromBackingInt(@intCast(2)),
    .depth_image = @fromBackingInt(@intCast(3)),
    .depth_view = @fromBackingInt(@intCast(4)),
    .extent = .{ .width = 1280, .height = 720 },
};

test "each barrier names its own image and no other" {
    const begin = pass.beginBarriers(target);
    try testing.expectEqual(target.hdr_image, begin[0].image);
    try testing.expectEqual(target.depth_image, begin[1].image);
    try testing.expectEqual(
        vk.ImageAspectFlags{ .color_bit = true },
        begin[0].subresource_range.aspect_mask,
    );
    try testing.expectEqual(
        vk.ImageAspectFlags{ .depth_bit = true },
        begin[1].subresource_range.aspect_mask,
    );

    const between = pass.prepassBarrier(target);
    try testing.expectEqual(target.depth_image, between[0].image);

    const end = pass.endBarriers(target);
    try testing.expectEqual(target.hdr_image, end[0].image);
    try testing.expectEqual(target.depth_image, end[1].image);
    try testing.expectEqual(
        vk.ImageAspectFlags{ .depth_bit = true },
        end[1].subresource_range.aspect_mask,
    );
}

test "the layout the pass leaves the target in is the one it was put into" {
    // Vulkan specification, VkImageMemoryBarrier2: oldLayout is either
    // VK_IMAGE_LAYOUT_UNDEFINED or the layout the image is currently in. The
    // begin barrier is what puts the HDR target in colour_attachment_optimal,
    // so the end barrier claiming anything else is a mismatch nothing else
    // here would show.
    const begin = pass.beginBarriers(target);
    const end = pass.endBarriers(target);

    try testing.expectEqual(begin[0].new_layout, end[0].old_layout);
    try testing.expectEqual(vk.ImageLayout.shader_read_only_optimal, end[0].new_layout);
    try testing.expectEqual(begin[1].new_layout, end[1].old_layout);
    try testing.expectEqual(pass.sampled_layout, end[1].new_layout);
}

test "the attachments the pass declares match the layouts the barriers set" {
    const colour = pass.colourAttachment(target, .{});
    const prepass_depth = pass.prepassDepthAttachment(target);
    const main_depth = pass.mainDepthAttachment(target);
    const begin = pass.beginBarriers(target);

    try testing.expectEqual(target.hdr_view, colour.image_view);
    try testing.expectEqual(begin[0].new_layout, colour.image_layout);
    try testing.expectEqual(target.depth_view, prepass_depth.image_view);
    try testing.expectEqual(begin[1].new_layout, prepass_depth.image_layout);
    try testing.expectEqual(prepass_depth.image_layout, main_depth.image_layout);
}

test "depth is stored between renderings and after shading, for compute to sample" {
    const colour = pass.colourAttachment(target, .{});
    const prepass_depth = pass.prepassDepthAttachment(target);
    const main_depth = pass.mainDepthAttachment(target);

    try testing.expectEqual(vk.AttachmentLoadOp.clear, colour.load_op);
    try testing.expectEqual(vk.AttachmentStoreOp.store, colour.store_op);
    try testing.expectEqual(vk.AttachmentLoadOp.clear, prepass_depth.load_op);
    try testing.expectEqual(vk.AttachmentStoreOp.store, prepass_depth.store_op);
    try testing.expectEqual(vk.AttachmentLoadOp.load, main_depth.load_op);
    try testing.expectEqual(vk.AttachmentStoreOp.store, main_depth.store_op);

    // The inter-rendering barrier orders the write-to-read handoff without a
    // layout transition.
    const between = pass.prepassBarrier(target)[0];
    try testing.expectEqual(between.old_layout, between.new_layout);
    try testing.expect(between.src_access_mask.depth_stencil_attachment_write_bit);
    try testing.expect(between.dst_access_mask.depth_stencil_attachment_read_bit);

    // After the main pass the depth writes reach compute's and the post pass's
    // sampled reads, and the next frame's first write waits for both.
    const after = pass.endBarriers(target)[1];
    try testing.expect(after.src_access_mask.depth_stencil_attachment_write_bit);
    try testing.expect(after.dst_stage_mask.compute_shader_bit);
    try testing.expect(after.dst_stage_mask.fragment_shader_bit);
    try testing.expect(after.dst_access_mask.shader_sampled_read_bit);
    const next = pass.beginBarriers(target)[1];
    try testing.expect(next.src_stage_mask.compute_shader_bit);
    try testing.expect(next.src_stage_mask.fragment_shader_bit);
}

test "the far plane is what the depth prepass clears to" {
    const depth = pass.prepassDepthAttachment(target);
    try testing.expectEqual(@as(f32, 1), depth.clear_value.depth_stencil.depth);

    const colour = pass.colourAttachment(target, .{ .clear_colour = .{ 0.1, 0.2, 0.3, 1 } });
    try testing.expectEqual(@as(f32, 0.2), colour.clear_value.color.float_32[1]);
}

test "the viewport covers the target and leaves depth unscaled" {
    const view = pass.viewport(target.extent);
    try testing.expectEqual(@as(f32, 1280), view.width);
    try testing.expectEqual(@as(f32, 720), view.height);
    try testing.expectEqual(@as(f32, 0), view.min_depth);
    try testing.expectEqual(@as(f32, 1), view.max_depth);

    const area = pass.scissor(target.extent);
    try testing.expectEqual(target.extent, area.extent);
    try testing.expectEqual(@as(i32, 0), area.offset.x);
}

test "the pass recording entry points are reached by the compiler" {
    _ = &pass.beginDepthPrepass;
    _ = &pass.endDepthPrepass;
    _ = &pass.beginMain;
    _ = &pass.endMain;
}
