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
const modes = [_]?gpu.MaterialRecord{
    .{ .alpha = .@"opaque", .mask = .{ .alpha_cutoff = 0.5, .factor_alpha = 1 } },
    null,
    .{ .alpha = .blend, .mask = .{ .alpha_cutoff = 0.5, .factor_alpha = 1 } },
};

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

// Never dereferenced. `planRecording` reads a batch's material index, cull mode
// and instance range and nothing else, so the pointer only has to address real
// storage; a mesh is a device object and there is none here.
var unread_mesh: gpu.Mesh = undefined;

fn batch(material_index: u32) gpu.RecordBatch {
    return .{
        .mesh = &unread_mesh,
        .material_index = material_index,
        .cull_mode = .{ .back_bit = true },
        .front_face = .counter_clockwise,
        .first_instance = 0,
        .instance_count = 1,
    };
}

// One live frame slot with room for the batches above, every descriptor
// configured, and a five-level chain for a look to be resolved against.
const ready_frames = [_]usize{4};
const ready: gpu.RecordState = .{
    .material_records = &modes,
    .frame_instance_counts = &ready_frames,
    .material_buffer_ready = true,
    .environment_ready = true,
    .bloom_levels = 5,
};

test "the background is planned at the batch the blended run opens with" {
    // The walk that validates the list is also what answers where the layers
    // meet, and this is the only place the two are checked against each other.
    // `backgroundSlot` is tested on a first-blended index it is handed, and
    // `validateRecordBatch` on one batch at a time; neither says that the index
    // reaching the slot is the one the list really has.
    const mixed = [_]gpu.RecordBatch{ batch(0), batch(0), batch(2), batch(2) };
    const planned = try gpu.planRecording(ready, .{
        .frame_index = 0,
        .batches = &mixed,
        .visible = mixed.len,
        .background = .environment,
        .bloom = null,
        .post = .{},
    });
    try testing.expectEqual(@as(?usize, 2), planned.background_slot);

    // The list the plan carries is the list that was checked. `record` takes no
    // second one, so this is what stands between the checks and the draws.
    try testing.expectEqual(&mixed[0], &planned.batches[0]);
    try testing.expectEqual(mixed.len, planned.batches.len);
    try testing.expectEqual(@as(usize, 0), planned.frame_index);
}

test "a list of one layer puts the background at that layer's edge" {
    const solid = [_]gpu.RecordBatch{ batch(0), batch(0) };
    const blended = [_]gpu.RecordBatch{ batch(2), batch(2) };

    // Nothing blends, so the background goes behind nothing and after
    // everything.
    const over_solid = try gpu.planRecording(ready, .{
        .frame_index = 0,
        .batches = &solid,
        .visible = solid.len,
        .background = .environment,
        .bloom = null,
        .post = .{},
    });
    try testing.expectEqual(@as(?usize, 2), over_solid.background_slot);

    // Everything blends, so the background is what the whole list composites
    // over.
    const under_blended = try gpu.planRecording(ready, .{
        .frame_index = 0,
        .batches = &blended,
        .visible = blended.len,
        .background = .environment,
        .bloom = null,
        .post = .{},
    });
    try testing.expectEqual(@as(?usize, 0), under_blended.background_slot);

    // And a list that asked for no background gets none wherever its layers
    // meet.
    const flat = try gpu.planRecording(ready, .{
        .frame_index = 0,
        .batches = &blended,
        .visible = blended.len,
        .background = .clear,
        .bloom = null,
        .post = .{},
    });
    try testing.expectEqual(@as(?usize, null), flat.background_slot);
}

test "an empty list with no background asks nothing of the renderer's state" {
    // A frame that draws neither a batch nor a background reads no material
    // array, no environment and no instance slice, so it plans against a
    // renderer that has none of them. This is the state a corpus walk is in
    // between two models.
    const bare: gpu.RecordState = .{
        .material_records = &.{},
        .frame_instance_counts = &.{},
        .material_buffer_ready = false,
        .environment_ready = false,
        .bloom_levels = 1,
    };
    const planned = try gpu.planRecording(bare, .{
        .frame_index = 7,
        .batches = &.{},
        .visible = 0,
        .background = .clear,
        .bloom = null,
        .post = .{},
    });
    try testing.expectEqual(@as(?usize, null), planned.background_slot);
    try testing.expectEqual(@as(usize, 0), planned.batches.len);

    // The background is the one draw that needs the environment without needing
    // a batch, and asking for it over an empty list is what says so.
    try testing.expectError(error.EnvironmentNotConfigured, gpu.planRecording(bare, .{
        .frame_index = 7,
        .batches = &.{},
        .visible = 0,
        .background = .environment,
        .bloom = null,
        .post = .{},
    }));
}

test "the resolved look is what the post constants are built from" {
    // Both halves of this are tested on their own: `bloomResolve` turns settings
    // into a look, and `postPushConstants` turns a look into a weight. Neither
    // says that the look the plan resolved is the one its constants carry, and a
    // plan that dropped it would compose a chain it had just recorded at a
    // weight of zero.
    const lit = try gpu.planRecording(ready, .{
        .frame_index = 0,
        .batches = &.{},
        .visible = 0,
        .background = .clear,
        .bloom = .{},
        .post = .{},
    });
    const look = lit.look orelse return error.TestExpectedLook;
    try testing.expectEqual(try gpu.bloomResolve(.{}, ready.bloom_levels), look);
    try testing.expectEqual(look.composite, lit.post_constants.bloom);
    try testing.expect(lit.post_constants.bloom > 0);

    // And a frame that asked for no chain composites nothing. Zero here is the
    // weight beside the pipeline that never samples the chain, not a chain
    // scaled to nothing.
    const dark = try gpu.planRecording(ready, .{
        .frame_index = 0,
        .batches = &.{},
        .visible = 0,
        .background = .clear,
        .bloom = null,
        .post = .{},
    });
    try testing.expectEqual(@as(?gpu.BloomLook, null), dark.look);
    try testing.expectEqual(@as(f32, 0), dark.post_constants.bloom);
}

test "a frame with nothing to draw still resolves the look it was given" {
    // Resolution is not skipped for a trivial frame. If it were, a look the
    // shader cannot evaluate would be reported on whichever later frame first
    // had a batch in it, which is a report against the wrong settings.
    try testing.expectError(error.InvalidScatter, gpu.planRecording(ready, .{
        .frame_index = 0,
        .batches = &.{},
        .visible = 0,
        .background = .clear,
        // The series the composite divides by does not converge at one.
        .bloom = .{ .scatter = 1 },
        .post = .{},
    }));

    try testing.expectError(error.InvalidExposure, gpu.planRecording(ready, .{
        .frame_index = 0,
        .batches = &.{},
        .visible = 0,
        .background = .clear,
        .bloom = null,
        .post = .{ .exposure = -1 },
    }));
}

test "a plan is refused for state the recording would index without checking" {
    const drawing = [_]gpu.RecordBatch{batch(0)};

    var missing_buffer = ready;
    missing_buffer.material_buffer_ready = false;
    try testing.expectError(error.MaterialBufferNotConfigured, gpu.planRecording(missing_buffer, .{
        .frame_index = 0,
        .batches = &drawing,
        .visible = drawing.len,
        .background = .clear,
        .bloom = null,
        .post = .{},
    }));

    var missing_environment = ready;
    missing_environment.environment_ready = false;
    try testing.expectError(error.EnvironmentNotConfigured, gpu.planRecording(missing_environment, .{
        .frame_index = 0,
        .batches = &drawing,
        .visible = drawing.len,
        .background = .clear,
        .bloom = null,
        .post = .{},
    }));

    // The frame slot indexes the instance counts, and the count it finds is the
    // bound every batch's instance range is checked against.
    try testing.expectError(error.FrameIndexOutOfRange, gpu.planRecording(ready, .{
        .frame_index = 1,
        .batches = &drawing,
        .visible = drawing.len,
        .background = .clear,
        .bloom = null,
        .post = .{},
    }));

    // The ordering rule reaches the plan, not just the single-batch check.
    const out_of_order = [_]gpu.RecordBatch{ batch(2), batch(0) };
    try testing.expectError(error.SolidBatchAfterBlended, gpu.planRecording(ready, .{
        .frame_index = 0,
        .batches = &out_of_order,
        .visible = out_of_order.len,
        .background = .environment,
        .bloom = null,
        .post = .{},
    }));
}

test "the three glTF alpha modes collapse onto the two pipelines" {
    // MASK has no pipeline of its own: a masked fragment either survives the
    // cutoff and behaves exactly like an opaque one or is discarded.
    try testing.expectEqual(gpu.PipelineMode.solid, gpu.pipelineModeFor(.@"opaque"));
    try testing.expectEqual(gpu.PipelineMode.solid, gpu.pipelineModeFor(.mask));
    try testing.expectEqual(gpu.PipelineMode.blended, gpu.pipelineModeFor(.blend));
}

test "a mesh draws through the pipeline its own streams call for" {
    // Each axis is the only thing standing between a mesh and a wrong draw. The
    // skinned pipeline is the only one declaring binding 1 and the only entry
    // point reading the joint array, so missing it draws the bind pose; the
    // second-UV pipeline is the only one declaring binding 3, so missing it
    // leaves a slot on set 1 sampling zeros.
    try testing.expectEqual(
        gpu.SceneVariant{ .skinned = false, .uv1 = false, .colour = false },
        gpu.sceneVariantFor(.{}),
    );
    try testing.expectEqual(
        gpu.SceneVariant{ .skinned = true, .uv1 = false, .colour = false },
        gpu.sceneVariantFor(.{ .skinned = true }),
    );
    try testing.expectEqual(
        gpu.SceneVariant{ .skinned = false, .uv1 = true, .colour = false },
        gpu.sceneVariantFor(.{ .uv1 = true }),
    );
    try testing.expectEqual(
        gpu.SceneVariant{ .skinned = true, .uv1 = true, .colour = false },
        gpu.sceneVariantFor(.{ .skinned = true, .uv1 = true }),
    );
}

test "every stream is an axis of the variant, and none masks another" {
    // Section 3.9.2 puts COLOR_0 in the base-colour product, so it selects an
    // entry point of its own rather than being carried and ignored: a mesh with
    // the stream draws through a different pipeline from one without.
    try testing.expect(!std.meta.eql(
        gpu.sceneVariantFor(.{}),
        gpu.sceneVariantFor(.{ .colour = true }),
    ));

    for (0..8) |bits| {
        const streams: res.VertexStreams = @bitCast(@as(u3, @intCast(bits)));
        const variant = gpu.sceneVariantFor(streams);
        try testing.expectEqual(streams.skinned, variant.skinned);
        try testing.expectEqual(streams.uv1, variant.uv1);
        try testing.expectEqual(streams.colour, variant.colour);
        // The vertex input a pipeline of this variant declares is the mesh's
        // own set of streams, which is what keeps `Mesh.bind` and the pipeline
        // naming the same bindings.
        try testing.expectEqual(streams, variant.streams());
    }
}

test "a variant's vertex input declares the second UV stream exactly when it asks for it" {
    // Why `SceneVariant.streams` exists: the vertex input and the entry point at
    // pipeline creation are built from this one value, so they cannot name
    // different sets of bindings. Which entry point actually reads location 7 is
    // settled against the compiler's reflection in `shader_reflection.zig`; what
    // is checked here is that the input behind it declares the attribute.
    const uv1_location = gpu.GpuUv1Vertex.attribute_descriptions[0].location;
    for (0..8) |bits| {
        const streams: res.VertexStreams = @bitCast(@as(u3, @intCast(bits)));
        const declared = gpu.sceneVariantFor(streams).streams();

        const input = gpu.pipelineVertexInput(declared);
        const has_uv1 = for (input.declaredAttributes()) |attribute| {
            if (attribute.location == uv1_location) break true;
        } else false;
        try testing.expectEqual(streams.uv1, has_uv1);
    }
}

test "every variant and mode lands on its own slot in the pipeline table" {
    // The creation loop fills the array in index order and its rollback walks
    // the built prefix, so the whole table rests on this being a bijection. Both
    // failures it guards against are startup failures on a device and reach no
    // test otherwise: a collision leaves one slot never written, and an index
    // past the end writes off the array.
    var filled: [gpu.scene_pipeline_count]bool = @splat(false);
    for ([_]gpu.SceneVariant{
        .{ .skinned = false, .uv1 = false, .colour = false },
        .{ .skinned = false, .uv1 = false, .colour = true },
        .{ .skinned = false, .uv1 = true, .colour = false },
        .{ .skinned = false, .uv1 = true, .colour = true },
        .{ .skinned = true, .uv1 = false, .colour = false },
        .{ .skinned = true, .uv1 = false, .colour = true },
        .{ .skinned = true, .uv1 = true, .colour = false },
        .{ .skinned = true, .uv1 = true, .colour = true },
    }) |variant| {
        for (gpu.scene_modes) |mode| {
            const index = gpu.scenePipelineIndex(variant, mode);
            try testing.expect(index < gpu.scene_pipeline_count);
            try testing.expect(!filled[index]);
            filled[index] = true;
        }
    }
    for (filled) |slot| try testing.expect(slot);

    // The table's axis is not the whole enum. The background is a pipeline of
    // its own, and a row of it here would be four more entries that no draw can
    // ask for: `modeFor` maps every glTF alpha mode onto the two above.
    try testing.expectEqual(@as(usize, 2), gpu.scene_modes.len);
    for (gpu.scene_modes) |mode| try testing.expect(mode != .background);
}

test "the four variants take four different slots on the vertex axis" {
    // A pipeline per variant is only worth building if the variants reach
    // different entry points, and this is the half of that the module owns: the
    // index a variant's vertex stage is read out of. Two axes collapsing onto
    // one slot would build pipelines that differ in vertex input alone, which
    // is a shader reading a binding its input does not declare.
    //
    // Which names sit in those slots is the table's property and is checked
    // where the table is authored.
    var seen: [gpu.scene_variants]bool = @splat(false);
    for ([_]res.VertexStreams{
        .{},
        .{ .skinned = true },
        .{ .uv1 = true },
        .{ .skinned = true, .uv1 = true },
    }) |streams| {
        const index = gpu.sceneVariantIndex(gpu.sceneVariantFor(streams));
        try testing.expect(index < gpu.scene_variants);
        try testing.expect(!seen[index]);
        seen[index] = true;
    }
}

test "a variant's vertex slot is the one its pipeline is built into" {
    // The two indices are one arithmetic, and this is what says so. A vertex
    // entry read out of one order and a pipeline written into another is every
    // draw transformed by the wrong variant with nothing reporting it.
    for ([_]gpu.SceneVariant{
        .{ .skinned = false, .uv1 = false, .colour = false },
        .{ .skinned = false, .uv1 = false, .colour = true },
        .{ .skinned = false, .uv1 = true, .colour = false },
        .{ .skinned = false, .uv1 = true, .colour = true },
        .{ .skinned = true, .uv1 = false, .colour = false },
        .{ .skinned = true, .uv1 = false, .colour = true },
        .{ .skinned = true, .uv1 = true, .colour = false },
        .{ .skinned = true, .uv1 = true, .colour = true },
    }) |variant| {
        for (gpu.scene_modes, 0..) |mode, offset| {
            try testing.expectEqual(
                gpu.sceneVariantIndex(variant) * gpu.scene_modes.len + offset,
                gpu.scenePipelineIndex(variant, mode),
            );
        }
    }
}

test "the plan carries the visible prefix to draw and the whole list to bake" {
    // The two lists a frame records, and the only place the split is checked.
    // Everything after the boundary reaches the bake and nothing else.
    const all = [_]gpu.RecordBatch{ batch(0), batch(2), batch(0), batch(2) };
    const planned = try gpu.planRecording(ready, .{
        .frame_index = 0,
        .batches = &all,
        .visible = 2,
        .background = .clear,
        .bloom = null,
        .post = .{},
    });

    try testing.expectEqual(@as(usize, 2), planned.batches.len);
    try testing.expectEqual(all.len, planned.casters.len);
    // Both name the caller's storage rather than a copy, and the visible list is
    // the front of the other one.
    try testing.expectEqual(&all[0], &planned.batches[0]);
    try testing.expectEqual(&all[0], &planned.casters[0]);
}

test "the layer rule is checked in the visible prefix and not behind it" {
    // A solid batch after a blended one cannot be composited, so the prefix
    // refuses it. Past the boundary the bake writes depth and blends nothing,
    // and the culled half arrives in whatever order culling left it.
    const behind = [_]gpu.RecordBatch{ batch(0), batch(2), batch(0) };
    const planned = try gpu.planRecording(ready, .{
        .frame_index = 0,
        .batches = &behind,
        // Only the first batch is on screen, so the blended-then-solid pair sits
        // entirely in the culled half.
        .visible = 1,
        .background = .clear,
        .bloom = null,
        .post = .{},
    });
    try testing.expectEqual(@as(usize, 1), planned.batches.len);

    // The same list with the boundary moved past the offending pair is refused,
    // which is what says the rule is about the prefix and not about the order
    // the batches happen to be in.
    try testing.expectError(error.SolidBatchAfterBlended, gpu.planRecording(ready, .{
        .frame_index = 0,
        .batches = &behind,
        .visible = behind.len,
        .background = .clear,
        .bloom = null,
        .post = .{},
    }));
}

test "a culled batch is still checked for what the bake indexes" {
    // The bake reads a material record for every caster and addresses the same
    // instance ring, so being off screen exempts a batch from the layer rule and
    // from nothing else.
    const unconfigured = [_]gpu.RecordBatch{ batch(0), batch(1) };
    try testing.expectError(error.MaterialNotConfigured, gpu.planRecording(ready, .{
        .frame_index = 0,
        .batches = &unconfigured,
        .visible = 1,
        .background = .clear,
        .bloom = null,
        .post = .{},
    }));

    var past_the_ring = [_]gpu.RecordBatch{ batch(0), batch(0) };
    past_the_ring[1].first_instance = 4;
    try testing.expectError(error.InstanceRangeOutOfBounds, gpu.planRecording(ready, .{
        .frame_index = 0,
        .batches = &past_the_ring,
        .visible = 1,
        .background = .clear,
        .bloom = null,
        .post = .{},
    }));
}

test "the background is placed against the visible count and not the list length" {
    // The background sits between the layers of the pass that draws it, and that
    // pass ends at the boundary. Placing it at the end of the whole list would
    // put it behind batches no camera pass records.
    const solid_then_culled = [_]gpu.RecordBatch{ batch(0), batch(0), batch(0) };
    const planned = try gpu.planRecording(ready, .{
        .frame_index = 0,
        .batches = &solid_then_culled,
        .visible = 1,
        .background = .environment,
        .bloom = null,
        .post = .{},
    });
    // One visible solid batch and no blended one, so the background follows it.
    try testing.expectEqual(@as(?usize, 1), planned.background_slot);
}

test "a visible count past the list is refused" {
    const drawn = [_]gpu.RecordBatch{batch(0)};
    try testing.expectError(error.VisibleCountOutOfRange, gpu.planRecording(ready, .{
        .frame_index = 0,
        .batches = &drawn,
        .visible = drawn.len + 1,
        .background = .clear,
        .bloom = null,
        .post = .{},
    }));
}
