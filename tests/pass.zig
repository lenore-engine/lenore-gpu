const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;
const pass = gpu.MainPass;

// Handles are only compared here, never dereferenced, so distinct synthetic
// ones are enough to tell the two attachments apart.
const target: gpu.MainPassTarget = .{
    .hdr_image = @enumFromInt(1),
    .hdr_view = @enumFromInt(2),
    .depth_image = @enumFromInt(3),
    .depth_view = @enumFromInt(4),
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

    const end = pass.endBarriers(target);
    try testing.expectEqual(target.hdr_image, end[0].image);
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
}

test "the attachments the pass declares match the layouts the barriers set" {
    const colour = pass.colourAttachment(target, .{});
    const depth = pass.depthAttachment(target);
    const begin = pass.beginBarriers(target);

    try testing.expectEqual(target.hdr_view, colour.image_view);
    try testing.expectEqual(begin[0].new_layout, colour.image_layout);
    try testing.expectEqual(target.depth_view, depth.image_view);
    try testing.expectEqual(begin[1].new_layout, depth.image_layout);
}

test "colour is kept and depth is thrown away" {
    // Depth exists only within the pass, which is what allows the image to be
    // lazily allocated. Storing it would silently cost memory bandwidth that
    // nothing reads.
    const colour = pass.colourAttachment(target, .{});
    const depth = pass.depthAttachment(target);

    try testing.expectEqual(vk.AttachmentLoadOp.clear, colour.load_op);
    try testing.expectEqual(vk.AttachmentStoreOp.store, colour.store_op);
    try testing.expectEqual(vk.AttachmentLoadOp.clear, depth.load_op);
    try testing.expectEqual(vk.AttachmentStoreOp.dont_care, depth.store_op);

    // Nothing samples depth, so the end of the pass has one barrier and it is
    // the colour one. A depth barrier here would be ordering a read that does
    // not happen.
    try testing.expectEqual(@as(usize, 1), pass.endBarriers(target).len);
}

test "the far plane is what the depth attachment clears to" {
    const depth = pass.depthAttachment(target);
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
    _ = &pass.begin;
    _ = &pass.end;
}
