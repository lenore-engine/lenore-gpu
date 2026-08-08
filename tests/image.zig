const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;

test "a depth view carries the depth aspect and never the colour one" {
    // Vulkan specification, VkImageSubresourceRange: the aspect names which
    // part of the image the view addresses. Depth read through a colour aspect
    // is not a view of anything, and the pair is what decides it.
    const colour = gpu.ImageKind.colour.aspect();
    const depth = gpu.ImageKind.depth.aspect();

    try testing.expectEqual(vk.ImageAspectFlags{ .color_bit = true }, colour);
    try testing.expectEqual(vk.ImageAspectFlags{ .depth_bit = true }, depth);
    try testing.expect(!depth.color_bit);
    try testing.expect(!colour.depth_bit);
}

test "a cube map is six layers, cube-compatible, and viewed as a cube" {
    // The three properties have to agree or the image is unusable: Vulkan
    // refuses a cube view of an image created without the flag, and a view
    // asking for six layers of a one-layer image. Pinning them together is what
    // makes a half-applied shape a failing test rather than a device error.
    const flat = gpu.ImageShape.texture_2d;
    const cube = gpu.ImageShape.cube;

    try testing.expectEqual(@as(u32, 1), flat.layerCount());
    try testing.expectEqual(@as(u32, 6), cube.layerCount());

    try testing.expectEqual(vk.ImageCreateFlags{}, flat.createFlags());
    try testing.expectEqual(vk.ImageCreateFlags{ .cube_compatible_bit = true }, cube.createFlags());

    try testing.expectEqual(vk.ImageViewType.@"2d", flat.viewType());
    try testing.expectEqual(vk.ImageViewType.cube, cube.viewType());
}

test "each usage is refused by the feature that would carry it" {
    // One usage at a time against a feature set holding everything else, so a
    // check wired to the wrong feature shows up as the wrong usage passing.
    const all: vk.FormatFeatureFlags = .{
        .transfer_src_bit = true,
        .transfer_dst_bit = true,
        .sampled_image_bit = true,
        .color_attachment_bit = true,
        .depth_stencil_attachment_bit = true,
    };

    const cases = [_]struct { usage: vk.ImageUsageFlags, missing: vk.FormatFeatureFlags }{
        .{ .usage = .{ .transfer_src_bit = true }, .missing = .{ .transfer_src_bit = true } },
        .{ .usage = .{ .transfer_dst_bit = true }, .missing = .{ .transfer_dst_bit = true } },
        .{ .usage = .{ .sampled_bit = true }, .missing = .{ .sampled_image_bit = true } },
        .{ .usage = .{ .color_attachment_bit = true }, .missing = .{ .color_attachment_bit = true } },
        .{
            .usage = .{ .depth_stencil_attachment_bit = true },
            .missing = .{ .depth_stencil_attachment_bit = true },
        },
    };

    for (cases) |case| {
        try testing.expect(gpu.imageFormatSupports(case.usage, all));

        var without = all;
        without = without.subtract(case.missing);
        try testing.expect(!gpu.imageFormatSupports(case.usage, without));
    }
}

test "a usage the format does not name is not consulted" {
    // A depth format reports no colour attachment feature, and a depth
    // attachment does not ask for one.
    const depth_only: vk.FormatFeatureFlags = .{ .depth_stencil_attachment_bit = true };

    try testing.expect(gpu.imageFormatSupports(.{ .depth_stencil_attachment_bit = true }, depth_only));
    try testing.expect(!gpu.imageFormatSupports(.{ .sampled_bit = true }, depth_only));

    // Nothing asked for is nothing to refuse.
    try testing.expect(gpu.imageFormatSupports(.{}, .{}));
}

test "the HDR target's whole usage has to be supported at once" {
    // It is rendered to and then sampled, so a format offering only one of the
    // two is not a candidate. Checking usages one at a time would accept it.
    const render_only: vk.FormatFeatureFlags = .{ .color_attachment_bit = true };
    const both: vk.FormatFeatureFlags = .{ .color_attachment_bit = true, .sampled_image_bit = true };
    const hdr_usage: vk.ImageUsageFlags = .{ .color_attachment_bit = true, .sampled_bit = true };

    try testing.expect(!gpu.imageFormatSupports(hdr_usage, render_only));
    try testing.expect(gpu.imageFormatSupports(hdr_usage, both));
}

test "an attachment usage is one this module will create" {
    // The gate before any format is consulted. A depth attachment refused here
    // fails as an unsupported usage, which reads as a missing feature rather
    // than as a usage this module never accepted.
    try testing.expect(gpu.imageUsageSupported(.{ .depth_stencil_attachment_bit = true }));
    try testing.expect(gpu.imageUsageSupported(.{ .color_attachment_bit = true, .sampled_bit = true }));
    try testing.expect(gpu.imageUsageSupported(.{ .transfer_dst_bit = true, .sampled_bit = true }));

    // Nothing here allocates lazily or renders to a tile-local attachment, so
    // the bit that would ask for it is not accepted.
    try testing.expect(!gpu.imageUsageSupported(.{ .transient_attachment_bit = true }));
    try testing.expect(!gpu.imageUsageSupported(.{ .storage_bit = true }));
}
