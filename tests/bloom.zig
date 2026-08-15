const std = @import("std");
const vk = @import("vulkan");
const gpu = @import("lenore-gpu");

const testing = std.testing;

// The handles are not dereferenced by anything under test: a pipeline
// configuration is data, and what it names is checked rather than what it points
// at.
const any_layout: vk.PipelineLayout = .null_handle;
const any_module: vk.ShaderModule = .null_handle;
const any_image: vk.Image = .null_handle;
const chain_format: vk.Format = .r16g16b16a16_sfloat;

// A chain shader as the module now takes one. The names are deliberately not
// the ones the engine ships: what is under test is that a configuration carries
// through the names it was given, and a spelling shared with the real table
// would pass whether it did or not.
const chain_shader: gpu.BloomShader = .{
    .spirv = &.{},
    .vertex_entry = "aVertexEntry",
    .downsample_entry = "aDownsampleEntry",
    .upsample_entry = "anUpsampleEntry",
};

test "the chain's base is half the target, and never nothing" {
    try testing.expectEqual(
        vk.Extent2D{ .width = 960, .height = 540 },
        gpu.bloomBaseExtent(.{ .width = 1920, .height = 1080 }),
    );

    // Odd extents truncate rather than round up. A level larger than half its
    // source would have the taps of the pattern land inside one texel.
    try testing.expectEqual(
        vk.Extent2D{ .width = 640, .height = 360 },
        gpu.bloomBaseExtent(.{ .width = 1281, .height = 721 }),
    );

    // A window one texel wide still has a chain. Zero is not an extent an image
    // can be created with.
    try testing.expectEqual(
        vk.Extent2D{ .width = 1, .height = 1 },
        gpu.bloomBaseExtent(.{ .width = 1, .height = 1 }),
    );
}

test "the chain stops where a level would fall under eight texels" {
    // The property rather than a table of answers: the deepest level is at least
    // eight texels on both sides, and the level after it would not be.
    const bases = [_]vk.Extent2D{
        .{ .width = 960, .height = 540 },
        .{ .width = 640, .height = 360 },
        .{ .width = 8, .height = 8 },
        .{ .width = 16, .height = 9 },
        .{ .width = 1024, .height = 1024 },
    };

    for (bases) |base| {
        const depth = gpu.bloomChainDepth(base);
        try testing.expect(depth >= 1);
        try testing.expect(depth <= gpu.bloom_max_levels);

        const deepest = depth - 1;
        if (deepest > 0) {
            try testing.expect(gpu.mipExtent(base.width, deepest) >= 8);
            try testing.expect(gpu.mipExtent(base.height, deepest) >= 8);
        }
        if (depth < gpu.bloom_max_levels) {
            const beyond = depth;
            try testing.expect(
                gpu.mipExtent(base.width, beyond) < 8 or
                    gpu.mipExtent(base.height, beyond) < 8,
            );
        }
    }
}

test "a base too small to reduce is a chain of one level" {
    // Not an empty chain and not an error. The extract step still runs, the
    // upsample has nothing to do, and the post pass reads what the extract left.
    try testing.expectEqual(@as(u32, 1), gpu.bloomChainDepth(.{ .width = 8, .height = 4 }));
    try testing.expectEqual(@as(u32, 1), gpu.bloomChainDepth(.{ .width = 1, .height = 1 }));
}

test "the deepest chain is bounded by the level cap" {
    // A base larger than any device permits still stops at the cap, so the view
    // array the pass carries is never overrun.
    const huge: vk.Extent2D = .{ .width = 65536, .height = 65536 };
    try testing.expectEqual(gpu.bloom_max_levels, gpu.bloomChainDepth(huge));
}

test "the composite normalizes the series the upsample generates" {
    // The property the whole weighting rests on: level k reaches level zero
    // multiplied by scatter to the k, so a uniformly bright field summed over
    // the chain and scaled by the composite comes back at the intensity asked
    // for, whatever depth the window allowed.
    const settings: gpu.BloomSettings = .{ .scatter = 0.7, .intensity = 0.3 };

    for ([_]u32{ 1, 2, 5, 7, gpu.bloom_max_levels }) |levels| {
        const look = try gpu.bloomResolve(settings, levels);

        var summed: f32 = 0;
        var level: u32 = 0;
        while (level < levels) : (level += 1)
            summed += std.math.pow(f32, look.scatter, @floatFromInt(level));

        try testing.expectApproxEqRel(settings.intensity, summed * look.composite, 1e-5);
    }
}

test "a scatter of zero leaves the finest level alone" {
    // Nothing carries upward, so the composite is the intensity itself. This is
    // the boundary the series expression has to survive: its numerator and its
    // denominator are both one there.
    const look = try gpu.bloomResolve(.{ .scatter = 0, .intensity = 0.5 }, 6);
    try testing.expectApproxEqRel(@as(f32, 0.5), look.composite, 1e-6);
}

test "resolve rejects every look the shader cannot evaluate" {
    const base: gpu.BloomSettings = .{};

    var knee = base;
    // Zero collapses the ramp onto a step, which smoothstep answers by dividing
    // by the span.
    knee.knee = 0;
    try testing.expectError(error.InvalidKnee, gpu.bloomResolve(knee, 4));

    var scatter = base;
    // One does not decay, and the series it sums is zero over zero.
    scatter.scatter = 1;
    try testing.expectError(error.InvalidScatter, gpu.bloomResolve(scatter, 4));

    var negative_scatter = base;
    negative_scatter.scatter = -0.1;
    try testing.expectError(error.InvalidScatter, gpu.bloomResolve(negative_scatter, 4));

    var threshold = base;
    threshold.threshold = -1;
    try testing.expectError(error.InvalidThreshold, gpu.bloomResolve(threshold, 4));

    var minimum = base;
    // A contribution floor is a fraction, so above one it would amplify rather
    // than pass through.
    minimum.minimum = 1.5;
    try testing.expectError(error.InvalidMinimum, gpu.bloomResolve(minimum, 4));

    var intensity = base;
    intensity.intensity = std.math.nan(f32);
    try testing.expectError(error.InvalidIntensity, gpu.bloomResolve(intensity, 4));

    var knee_infinite = base;
    knee_infinite.knee = std.math.inf(f32);
    try testing.expectError(error.InvalidKnee, gpu.bloomResolve(knee_infinite, 4));
}

test "the defaults are the look the reference picture was read for" {
    // Khronos publishes a bloom render of EmissiveStrengthTest whose first cube
    // peaks at 0.9 and carries no halo, and whose second peaks at 1.8 and
    // carries a faint one. A threshold at or below 0.9 would put a halo on the
    // first; a ramp ending at or below 1.8 would give the second a full one.
    const defaults: gpu.BloomSettings = .{};
    try testing.expect(defaults.threshold > 0.9);
    try testing.expect(defaults.threshold + defaults.knee > 1.8);
    _ = try gpu.bloomResolve(defaults, 7);
}

test "the push block matches the layout slangc reported" {
    // Measured from the reflection JSON slangc emitted beside the words, not
    // derived from the field list. This holds the Zig side to those numbers, so
    // a field reordered here is caught without a device; holding a shader
    // against them needs the compiler's account of it, which is the job of
    // whatever supplies the words.
    const Push = gpu.BloomPushConstants;
    try testing.expectEqual(@as(usize, 0), @offsetOf(Push, "source_texel"));
    try testing.expectEqual(@as(usize, 8), @offsetOf(Push, "exposure"));
    try testing.expectEqual(@as(usize, 12), @offsetOf(Push, "threshold"));
    try testing.expectEqual(@as(usize, 16), @offsetOf(Push, "knee"));
    try testing.expectEqual(@as(usize, 20), @offsetOf(Push, "minimum"));
    try testing.expectEqual(@as(usize, 24), @offsetOf(Push, "weight"));
    try testing.expectEqual(@as(usize, 28), @offsetOf(Push, "extract"));
    try testing.expectEqual(@as(usize, 32), @sizeOf(Push));
}

test "the push range covers the block, in the stage that reads it" {
    const range = gpu.bloomPushConstantRange;
    try testing.expectEqual(@as(u32, 0), range.offset);
    try testing.expectEqual(@as(u32, @sizeOf(gpu.BloomPushConstants)), range.size);
    try testing.expect(range.stage_flags.fragment_bit);
    // Only the fragment stage reads it. The vertex stage generates its three
    // positions from the index and touches nothing in the block.
    try testing.expect(!range.stage_flags.vertex_bit);
}

test "the tap spacing is one over the level being read" {
    const texel = gpu.bloomTexelSize(.{ .width = 960, .height = 540 });
    try testing.expectApproxEqRel(@as(f32, 1.0 / 960.0), texel[0], 1e-6);
    try testing.expectApproxEqRel(@as(f32, 1.0 / 540.0), texel[1], 1e-6);
}

test "the chain declares one sampled binding, in the fragment stage" {
    try testing.expectEqual(@as(usize, 1), gpu.bloom_bindings.len);
    try testing.expectEqual(@as(u32, 0), gpu.bloom_bindings[0].slot);
    try testing.expectEqual(vk.DescriptorType.combined_image_sampler, gpu.bloom_bindings[0].kind);
    try testing.expect(gpu.bloom_bindings[0].stages.fragment_bit);
}

test "the two directions differ in the fragment stage and in nothing else" {
    const down = gpu.Bloom.downConfig(any_layout, any_module, chain_shader, chain_format);
    const up = gpu.Bloom.upConfig(any_layout, any_module, chain_shader, chain_format);

    // The upsample adds onto what the downsample left in the level. The
    // downsample writes every texel of its target and replaces it.
    try testing.expectEqual(gpu.PipelineMode.solid, down.mode);
    try testing.expectEqual(gpu.PipelineMode.additive, up.mode);

    // Neither reads a vertex buffer, and neither culls: a covering triangle has
    // no back to face away.
    try testing.expectEqual(0, down.vertex_input.binding_count);
    try testing.expectEqual(0, up.vertex_input.binding_count);
    try testing.expectEqual(vk.CullModeFlags{}, down.culling.fixed);
    try testing.expectEqual(vk.CullModeFlags{}, up.culling.fixed);

    // No depth attachment in either. Declaring one would demand a depth image
    // the chain does not have.
    try testing.expectEqual(@as(?vk.Format, null), down.formats.depth);
    try testing.expectEqual(@as(?vk.Format, null), up.formats.depth);
    try testing.expectEqual(chain_format, down.formats.colour.?);
    try testing.expectEqual(chain_format, up.formats.colour.?);

    // Both directions take the one vertex stage they were given, and each takes
    // its own fragment stage. Held against the supplied names rather than only
    // against each other: two configurations agreeing on the wrong name agree
    // just as well as on the right one.
    try testing.expectEqualStrings(
        std.mem.span(chain_shader.vertex_entry),
        std.mem.span(down.stages.vertex.entry_point),
    );
    try testing.expectEqualStrings(
        std.mem.span(chain_shader.vertex_entry),
        std.mem.span(up.stages.vertex.entry_point),
    );
    try testing.expectEqualStrings(
        std.mem.span(chain_shader.downsample_entry),
        std.mem.span(down.stages.fragment.?.entry_point),
    );
    try testing.expectEqualStrings(
        std.mem.span(chain_shader.upsample_entry),
        std.mem.span(up.stages.fragment.?.entry_point),
    );
}

test "the additive mode sums the source into the destination and reads no alpha" {
    const blend = gpu.pipelineBlendAttachment(.additive);
    try testing.expectEqual(vk.Bool32.true, blend.blend_enable);
    try testing.expectEqual(vk.BlendFactor.one, blend.src_color_blend_factor);
    try testing.expectEqual(vk.BlendFactor.one, blend.dst_color_blend_factor);
    try testing.expectEqual(vk.BlendOp.add, blend.color_blend_op);

    // The weight of a level is a uniform the shader applies, so it must not also
    // appear as a blend factor: two places scaling one summand is one of them
    // being applied twice.
    try testing.expect(blend.src_color_blend_factor != .constant_color);
    try testing.expect(blend.dst_color_blend_factor != .constant_color);
}

test "the additive mode has no depth behaviour to inherit" {
    const depth = gpu.pipelineDepthStencilState(.additive);
    try testing.expectEqual(vk.Bool32.false, depth.depth_test_enable);
    try testing.expectEqual(vk.Bool32.false, depth.depth_write_enable);
}

test "every level barrier names its own level" {
    // One image and one level in flight at a time. A barrier covering the whole
    // chain would transition the level being sampled beside the one being
    // written, which is the hazard the chain is walked level by level to avoid.
    for ([_]gpu.BloomLevelTransition{
        .discard_to_attachment,
        .attachment_to_sampled,
        .sampled_to_attachment,
    }) |transition| {
        const barrier = gpu.bloomLevelBarrier(any_image, 3, transition);
        try testing.expectEqual(@as(u32, 3), barrier.subresource_range.base_mip_level);
        try testing.expectEqual(@as(u32, 1), barrier.subresource_range.level_count);
        try testing.expectEqual(@as(u32, 1), barrier.subresource_range.layer_count);
        try testing.expect(barrier.subresource_range.aspect_mask.color_bit);
        try testing.expectEqual(vk.QUEUE_FAMILY_IGNORED, barrier.src_queue_family_index);
        try testing.expectEqual(vk.QUEUE_FAMILY_IGNORED, barrier.dst_queue_family_index);
    }
}

test "the first write of a level discards it and waits on last frame" {
    const barrier = gpu.bloomLevelBarrier(any_image, 0, .discard_to_attachment);

    // One chain serves every frame in flight, so this write follows the previous
    // frame's read of the same level and its write of it. The read needs the
    // execution half of the dependency and the write needs the access half,
    // which is why the source stage names both and the source access names only
    // the write.
    try testing.expect(barrier.src_stage_mask.fragment_shader_bit);
    try testing.expect(barrier.src_stage_mask.color_attachment_output_bit);
    try testing.expect(barrier.src_access_mask.color_attachment_write_bit);
    try testing.expect(!barrier.src_access_mask.shader_read_bit);

    // Undefined, because every texel of the level is about to be written by a
    // covering triangle. Preserving last frame's contents would be bandwidth
    // spent on values all of which are replaced.
    try testing.expectEqual(vk.ImageLayout.undefined, barrier.old_layout);
    try testing.expectEqual(vk.ImageLayout.color_attachment_optimal, barrier.new_layout);
}

test "a written level is handed to the sampler that reads it next" {
    const barrier = gpu.bloomLevelBarrier(any_image, 2, .attachment_to_sampled);
    try testing.expect(barrier.src_stage_mask.color_attachment_output_bit);
    try testing.expect(barrier.src_access_mask.color_attachment_write_bit);
    try testing.expect(barrier.dst_stage_mask.fragment_shader_bit);
    try testing.expect(barrier.dst_access_mask.shader_read_bit);
    try testing.expectEqual(vk.ImageLayout.color_attachment_optimal, barrier.old_layout);

    // The layout the main pass leaves its own target in, named once so that a
    // descriptor declaring it and a barrier producing it cannot differ.
    try testing.expectEqual(gpu.mainPassSampledLayout, barrier.new_layout);
}

test "the upsample's target is readable by the blend that composites onto it" {
    const barrier = gpu.bloomLevelBarrier(any_image, 1, .sampled_to_attachment);

    // A write after the read the downsample above made of this level, so the
    // source scope is execution only.
    try testing.expect(barrier.src_stage_mask.fragment_shader_bit);
    try testing.expectEqual(vk.AccessFlags2{}, barrier.src_access_mask);

    // The additive blend reads the destination before it writes it, so a
    // destination scope naming the write alone would under-synchronize the half
    // of the operation that reads.
    try testing.expect(barrier.dst_access_mask.color_attachment_read_bit);
    try testing.expect(barrier.dst_access_mask.color_attachment_write_bit);

    try testing.expectEqual(gpu.mainPassSampledLayout, barrier.old_layout);
    try testing.expectEqual(vk.ImageLayout.color_attachment_optimal, barrier.new_layout);
}
