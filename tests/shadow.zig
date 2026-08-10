const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");
const res = @import("lenore-resources");

const testing = std.testing;
const shadow = gpu.Shadow;

// A caster's pipeline is chosen from two independent facts, and the ordering
// below is not decorative: `ShadowPass.init` fills its array by walking the two
// axes in this order and asserts the index as it goes, so an index function that
// disagreed would be caught at creation. What it cannot catch is two variants
// mapping to the same slot, which would leave one pipeline built and never
// bound.
test "each caster variant has a pipeline slot of its own" {
    var seen: [4]bool = @splat(false);
    for ([_]bool{ false, true }) |skinned| {
        for ([_]bool{ false, true }) |masked| {
            const index = shadow.pipelineIndex(.{ .skinned = skinned, .masked = masked });
            try testing.expect(index < seen.len);
            try testing.expect(!seen[index]);
            seen[index] = true;
        }
    }
    for (seen) |filled| try testing.expect(filled);
}

test "the skinned axis and the masked axis are independent" {
    const plain = shadow.pipelineIndex(.{ .skinned = false, .masked = false });
    const skinned = shadow.pipelineIndex(.{ .skinned = true, .masked = false });
    const masked = shadow.pipelineIndex(.{ .skinned = false, .masked = true });
    const both = shadow.pipelineIndex(.{ .skinned = true, .masked = true });

    // Changing one axis moves the index by the same step whatever the other is.
    // An index built as `skinned + masked` would pass every distinctness check
    // above for three of the four and collide on the fourth.
    try testing.expectEqual(skinned - plain, both - masked);
    try testing.expectEqual(masked - plain, both - skinned);
}

// The three alpha modes reach the bake as three different answers, and the
// middle one is the reason the renderer stores the authored mode rather than the
// blend mode it maps to: MASK and OPAQUE collapse together for the main pass and
// must not here.
test "BLEND casts nothing, MASK casts through the fragment stage, OPAQUE does not" {
    try testing.expectEqual(@as(?shadow.Variant, null), shadow.casterVariant(.blend, false));
    try testing.expectEqual(@as(?shadow.Variant, null), shadow.casterVariant(.blend, true));

    const masked = shadow.casterVariant(.mask, false) orelse return error.MaskDoesNotCast;
    try testing.expect(masked.masked);
    const solid = shadow.casterVariant(.@"opaque", false) orelse return error.OpaqueDoesNotCast;
    try testing.expect(!solid.masked);
}

test "the mesh's own streams decide the skinned axis, not the alpha mode" {
    for ([_]res.MaterialInfo.Rendering.AlphaMode{ .@"opaque", .mask }) |alpha| {
        const rigid = shadow.casterVariant(alpha, false) orelse return error.DoesNotCast;
        const skinned = shadow.casterVariant(alpha, true) orelse return error.DoesNotCast;
        try testing.expect(!rigid.skinned);
        try testing.expect(skinned.skinned);
        // And the alpha axis is untouched by the streams.
        try testing.expectEqual(rigid.masked, skinned.masked);
    }
}

// The two numbers a masked caster is tested with come from two different places
// in the authored material, and pairing them wrongly is invisible in every
// material whose base colour factor is opaque white, which is most of them.
test "a material record takes the cutoff and the factor alpha from their own fields" {
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{},
        .factors = .{ .base_colour = .{ 1, 1, 1, 0.25 } },
        .rendering = .{ .alpha_mode = .mask, .alpha_cutoff = 0.75 },
    };

    const record = gpu.MaterialRecord.forMaterial(&info);
    try testing.expectEqual(@as(f32, 0.75), record.mask.alpha_cutoff);
    try testing.expectEqual(@as(f32, 0.25), record.mask.factor_alpha);
    try testing.expectEqual(res.MaterialInfo.Rendering.AlphaMode.mask, record.alpha);

    // Distinct values on purpose: swapped fields would agree at the default
    // cutoff of 0.5 against a factor alpha of 1.0 only by accident, and this
    // pair separates them.
    info.rendering.alpha_cutoff = 0.25;
    info.factors.base_colour[3] = 0.75;
    const swapped = gpu.MaterialRecord.forMaterial(&info);
    try testing.expectEqual(@as(f32, 0.25), swapped.mask.alpha_cutoff);
    try testing.expectEqual(@as(f32, 0.75), swapped.mask.factor_alpha);
}

test "the main pass sees two modes where the bake sees three" {
    const modes = [_]res.MaterialInfo.Rendering.AlphaMode{ .@"opaque", .mask, .blend };
    const expected = [_]gpu.PipelineMode{ .solid, .solid, .blended };

    for (modes, expected) |alpha, mode| {
        const record: gpu.MaterialRecord = .{
            .alpha = alpha,
            .mask = .{ .alpha_cutoff = 0.5, .factor_alpha = 1 },
        };
        try testing.expectEqual(mode, record.mode());
    }
    // MASK and OPAQUE agree for the main pass and disagree for the bake. That
    // pair is the whole reason the authored mode is what gets stored.
    try testing.expect(shadow.casterVariant(.mask, false).?.masked !=
        shadow.casterVariant(.@"opaque", false).?.masked);
}

// One image serves every frame in flight, so the bake is a write against the
// previous frame's read of the same image. Both halves of that are checked:
// getting the direction backwards produces a barrier that orders nothing and
// that no validation layer reports.
test "the bake is ordered after the previous frame's sampling and before the next" {
    const handle: vk.Image = .null_handle;

    const opening = shadow.beginBarriers(handle)[0];
    try testing.expect(opening.src_stage_mask.fragment_shader_bit);
    try testing.expect(opening.src_access_mask.shader_read_bit);
    try testing.expect(opening.dst_access_mask.depth_stencil_attachment_write_bit);
    try testing.expectEqual(vk.ImageLayout.depth_attachment_optimal, opening.new_layout);

    const closing = shadow.endBarriers(handle)[0];
    try testing.expect(closing.src_access_mask.depth_stencil_attachment_write_bit);
    try testing.expect(closing.dst_stage_mask.fragment_shader_bit);
    try testing.expect(closing.dst_access_mask.shader_read_bit);

    // The layout the pass leaves the map in is the layout the descriptor was
    // written with, and they are named in two files.
    try testing.expectEqual(gpu.mainPassSampledLayout, closing.new_layout);
    try testing.expectEqual(opening.new_layout, closing.old_layout);

    // Depth alone. A colour aspect here names a plane the image does not have.
    try testing.expect(opening.subresource_range.aspect_mask.depth_bit);
    try testing.expect(!opening.subresource_range.aspect_mask.color_bit);
}

// Unlike the camera's depth attachment, which is written and thrown away, this
// one is the pass's entire output.
test "the map is cleared to the far plane and kept" {
    const info = shadow.depthAttachment(.null_handle);
    try testing.expectEqual(vk.AttachmentLoadOp.clear, info.load_op);
    try testing.expectEqual(vk.AttachmentStoreOp.store, info.store_op);

    // The far plane, so an untouched texel holds a caster infinitely far away
    // and every comparison against it passes. Cleared to zero it would report
    // the whole scene as shadowed, which is the same picture as a bake that
    // failed and is why this is worth pinning.
    try testing.expectEqual(@as(f32, 1), info.clear_value.depth_stencil.depth);
}
