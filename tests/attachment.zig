const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;

const depth_feature: vk.FormatFeatureFlags = .{ .depth_stencil_attachment_bit = true };
const hdr_features: vk.FormatFeatureFlags = .{
    .color_attachment_bit = true,
    .sampled_image_bit = true,
};
const none: vk.FormatFeatureFlags = .{};

test "the first candidate the device carries is the one taken" {
    const candidates = [_]vk.Format{ .d32_sfloat, .x8_d24_unorm_pack32, .d16_unorm };

    try testing.expectEqual(
        vk.Format.d32_sfloat,
        gpu.attachmentFirstSupported(
            &candidates,
            &.{ depth_feature, depth_feature, depth_feature },
            gpu.Attachment.depth_usage,
        ),
    );

    // The order is a preference, not a fallback chain that only the last entry
    // can end: dropping the first has to reach the second, not the third.
    try testing.expectEqual(
        vk.Format.x8_d24_unorm_pack32,
        gpu.attachmentFirstSupported(
            &candidates,
            &.{ none, depth_feature, depth_feature },
            gpu.Attachment.depth_usage,
        ),
    );
    try testing.expectEqual(
        vk.Format.d16_unorm,
        gpu.attachmentFirstSupported(
            &candidates,
            &.{ none, none, depth_feature },
            gpu.Attachment.depth_usage,
        ),
    );
}

test "a device carrying no candidate names nothing rather than the last one" {
    try testing.expectEqual(
        @as(?vk.Format, null),
        gpu.attachmentFirstSupported(
            &gpu.Attachment.depth_candidates,
            &.{ none, none, none },
            gpu.Attachment.depth_usage,
        ),
    );
}

test "an HDR candidate has to carry both of its usages" {
    // Rendered to and then sampled. A format offering only the attachment
    // feature is skipped, which is the case a per-usage check would let past.
    const render_only: vk.FormatFeatureFlags = .{ .color_attachment_bit = true };

    try testing.expectEqual(
        vk.Format.r16g16b16a16_sfloat,
        gpu.attachmentFirstSupported(
            &gpu.Attachment.hdr_candidates,
            &.{ render_only, hdr_features },
            gpu.Attachment.hdr_usage,
        ),
    );
}

test "the preferred formats are the narrow ones" {
    // Bandwidth is the reason the order is what it is, so the order is the
    // thing worth pinning: 32 bits per pixel ahead of 64, and depth without a
    // stencil that nothing uses.
    try testing.expectEqual(vk.Format.b10g11r11_ufloat_pack32, gpu.Attachment.hdr_candidates[0]);
    try testing.expectEqual(vk.Format.r16g16b16a16_sfloat, gpu.Attachment.hdr_candidates[1]);
    try testing.expectEqual(vk.Format.d32_sfloat, gpu.Attachment.depth_candidates[0]);

    for (gpu.Attachment.depth_candidates) |format| {
        try testing.expect(format != .d32_sfloat_s8_uint);
        try testing.expect(format != .d24_unorm_s8_uint);
    }
}

test "depth is never sampled and colour always is" {
    // The pair the pass depends on: depth without the sampled bit is what lets
    // the attachment be discarded, and the HDR target without it could not be
    // tonemapped.
    try testing.expect(!gpu.Attachment.depth_usage.sampled_bit);
    try testing.expect(gpu.Attachment.hdr_usage.sampled_bit);
    try testing.expect(gpu.Attachment.hdr_usage.color_attachment_bit);
    try testing.expect(gpu.Attachment.depth_usage.depth_stencil_attachment_bit);
}

test "the attachment constructors are reached by the compiler" {
    _ = &gpu.Attachment.createDepth;
    _ = &gpu.Attachment.createHdr;
    _ = &gpu.attachmentDepthFormat;
    _ = &gpu.attachmentHdrFormat;
}
