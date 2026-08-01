const std = @import("std");
const platform = @import("lenore-platform");
const gpu = @import("lenore-gpu");
const res = @import("lenore-resources");

const checking = std.debug.runtime_safety;
var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

// The example's own numbers, not engine policy and not measured. A consumer has
// to choose block sizes, and this is what choosing looks like.
const memory_config: gpu.MemoryConfig = .{
    .device_buffer_block_size = 64 << 20,
    .device_image_block_size = 64 << 20,
    .upload_buffer_block_size = 16 << 20,
    .readback_buffer_block_size = 8 << 20,
};

pub fn main(process: std.process.Init.Minimal) !void {
    const gpa = if (checking) debug_allocator.allocator() else std.heap.smp_allocator;
    defer if (checking) {
        if (debug_allocator.deinit() == .leak) std.log.err("memory leaked", .{});
    };

    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var host: platform.Platform = try .init();
    defer host.deinit();

    var window = try host.createWindow(.{ .width = 1280, .height = 720 }, "lenore-gpu core");
    defer window.deinit();

    var context: gpu.Context = try .init(gpa, "lenore-gpu core", window.nativeHandles());
    defer context.deinit();
    // What the device selector actually chose. Compilation cannot show any of
    // it, and a run that only prints the device name leaves the families and the
    // optional capabilities unconfirmed.
    std.log.info("device: {s}", .{context.deviceName()});
    std.log.info("queue families: graphics {d}, present {d}", .{
        context.graphics_queue.family,
        context.present_queue.family,
    });
    std.log.info("optional capabilities: memory budget {}, pipeline statistics {}", .{
        context.memory_budget_enabled,
        context.pipeline_statistics_enabled,
    });
    std.log.info("limits: max buffer size {d} MiB, max sampler anisotropy {d:.1}", .{
        context.max_buffer_size >> 20,
        context.max_sampler_anisotropy,
    });
    // The mip cap in image.zig and ktx2.zig is a chosen 16 levels, which
    // describes extents up to 32768. This prints what the device would actually
    // allow, so the relationship between the two is measured rather than
    // assumed. Vulkan specification, Limits, Required Limits table:
    // maxImageDimension2D is at least 4096 on any conformant implementation,
    // which is 13 levels.
    const image_limit = context.properties.limits.max_image_dimension_2d;
    std.log.info("limits: max 2D image {d}, so {d} mip levels", .{
        image_limit,
        std.math.log2_int(u32, image_limit) + 1,
    });

    var allocator: gpu.MemoryAllocator = try .init(&context, gpa, io, memory_config);
    defer if (allocator.deinit() == .leak) std.log.err("gpu memory leaked", .{});

    // Upload memory is persistently mapped and coherent, so the write below needs
    // no flush. Ordering against the GPU read is what the submit provides: the
    // host write completes before vkQueueSubmit2, and Vulkan specification,
    // Host Write Ordering Guarantees, makes that write visible to the copy
    // without a barrier.
    // Large enough for a 2048-square BC7 mip chain. BC7 spends one byte per
    // texel, so the base level is 4 MiB and the chain converges on four thirds
    // of that, a little under 5.6 MiB. A smaller arena is not an error: reserve
    // reports that the request never fits, as distinct from not fitting right
    // now.
    var staging: gpu.StagingArena = try .init(&context, &allocator, 8 << 20);
    defer staging.deinit();

    // Two reservations, so the second one's offset is what makes the copy region
    // explicit rather than incidental.
    const filler = try staging.reserve(64, 4);
    @memset(filler.bytes, 0x5A);
    const vertex_bytes = try staging.reserve(256, 4);
    @memset(vertex_bytes.bytes, 0xA5);

    // One 4x4 RGBA level, tightly packed. A block-compressed format would size
    // its levels by block count instead, which is the container's job and not
    // the image's.
    const texel_size = 4;
    const texture_extent = 4;
    const texture_bytes = try staging.reserve(
        texture_extent * texture_extent * texel_size,
        texel_size,
    );
    @memset(texture_bytes.bytes, 0xFF);

    var vertices: gpu.Buffer = try .init(
        &context,
        &allocator,
        1024,
        // A transfer source as well, so the round trip below can read it back.
        // A vertex buffer needs no such bit to be drawn from; this one earns it
        // by being copied out of.
        .{ .transfer_dst_bit = true, .transfer_src_bit = true, .vertex_buffer_bit = true },
        .device,
    );
    defer vertices.deinit();

    // Cold setup: one copy, submitted on its own and waited for. A frame loop
    // would batch this instead of idling the queue per command buffer.
    var setup_pool: gpu.OneShotPool = try .init(&context);
    defer setup_pool.deinit(&context);

    // One copy, recorded and submitted on its own. Everything below goes through
    // the upload transaction instead; this stays because it is the only direct
    // use of a copy region with a non-zero source offset.
    const setup = try gpu.beginOneShot(&context, setup_pool.handle);
    try vertices.recordCopyFrom(staging.source(), setup, .{
        .source_offset = vertex_bytes.offset,
        .size = vertex_bytes.bytes.len,
    });
    try gpu.submitOneShotAndWait(&context, setup_pool.handle, setup);

    // The round trip: host write into mapped upload memory, copy to device-local
    // memory, copy back into mapped readback memory, host read. It verifies the
    // bytes, which a validation layer does not: that reports misuse of the API,
    // not what the API moved.
    var readback: gpu.Buffer = try .init(
        &context,
        &allocator,
        1024,
        .{ .transfer_dst_bit = true },
        .readback,
    );
    defer readback.deinit();

    const verify = try gpu.beginOneShot(&context, setup_pool.handle);
    try readback.recordCopyFrom(&vertices, verify, .{ .size = vertex_bytes.bytes.len });
    try gpu.submitOneShotAndWait(&context, setup_pool.handle, verify);

    const returned = readback.mapped().?[0..vertex_bytes.bytes.len];
    var intact = true;
    for (returned) |byte| intact = intact and byte == 0xA5;
    std.log.info("readback: {d} bytes, contents intact {}", .{ returned.len, intact });

    var swapchain: gpu.Swapchain = try .init(&context, gpa, .{ .width = 1280, .height = 720 }, .fifo);
    defer swapchain.deinit();

    var frame: gpu.Frame = try .init(&context);
    defer frame.deinit(&context);

    // The texture cache records one upload per fallback, so it needs a
    // submission of its own before anything can bind them.
    const cache_setup = try gpu.beginOneShot(&context, setup_pool.handle);
    var textures: gpu.TextureCache = try .init(&context, &allocator, gpa, &staging, cache_setup);
    defer if (textures.deinit() == .leak) std.log.err("texture references outstanding", .{});
    try gpu.submitOneShotAndWait(&context, setup_pool.handle, cache_setup);

    var storage: gpu.ResourceStorage = .empty;
    defer storage.deinit(gpa);

    // A KTX2 file to load, if one was named. Without it every slot binds its
    // neutral fallback, which is the path a material without textures takes.
    var arguments: std.process.Args.Iterator = .init(process.args);
    _ = arguments.skip();
    const texture_path = arguments.next();
    const ktx2_bytes: ?[]u8 = if (texture_path) |path|
        try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20))
    else
        null;
    defer if (ktx2_bytes) |owned| gpa.free(owned);

    // A quad carrying every optional stream, so one batch exercises the
    // skinning, colour, second-UV and morph paths as well as the mandatory one.
    const quad = [_]res.Vertex3D{
        .{ .position = .{ -1, -1, 0 }, .normal = .{ 0, 0, 1 }, .uv = .{ 0, 0 }, .tangent = .{ 1, 0, 0, 1 } },
        .{ .position = .{ 1, -1, 0 }, .normal = .{ 0, 0, 1 }, .uv = .{ 1, 0 }, .tangent = .{ 1, 0, 0, 1 } },
        .{ .position = .{ 1, 1, 0 }, .normal = .{ 0, 0, 1 }, .uv = .{ 1, 1 }, .tangent = .{ 1, 0, 0, 1 } },
        .{ .position = .{ -1, 1, 0 }, .normal = .{ 0, 0, 1 }, .uv = .{ 0, 1 }, .tangent = .{ 1, 0, 0, 1 } },
    };
    const quad_indices = [_]u16{ 0, 1, 2, 2, 3, 0 };
    const morph_positions = [_]f32{0.0} ** (quad.len * 2 * 3);

    // The transaction: every copy into one command buffer, submitted once. A
    // failure anywhere below rolls the whole thing back, command buffer first.
    var batch = try gpu.UploadBatch.begin(
        gpa,
        &context,
        &allocator,
        &storage,
        &textures,
        &staging,
        setup_pool.handle,
    );
    var uploaded: gpu.Uploaded = uploaded: {
        errdefer batch.deinit();

        const mesh_handle = try batch.addMesh(u16, .{
            .vertices = &quad,
            .indices = &quad_indices,
            .streams = .{ .skinned = true, .colour = true, .uv1 = true },
            .morph = .{ .positions = &morph_positions, .target_count = 2 },
        });

        const set_handle = try batch.addTextureSet(.{
            .base_colour = if (ktx2_bytes) |bytes| .{
                .key = texture_path.?,
                .ktx2_bytes = bytes,
            } else null,
        });

        const mesh = storage.mesh(mesh_handle).?;
        std.log.info("mesh: {d} vertices, {d} indices, {d} morph targets, radius {d:.3}", .{
            mesh.vertex_count,
            mesh.index_count,
            mesh.morph_target_count,
            mesh.bounds.sphere.radius,
        });

        const set = storage.textureSet(set_handle).?;
        std.log.info("texture set: base colour {d}x{d} with {d} mips, normal {d}x{d}", .{
            set.base_colour.width,
            set.base_colour.height,
            set.base_colour.mip_levels,
            set.normal.width,
            set.normal.height,
        });

        break :uploaded try batch.finish();
    };
    defer uploaded.deinit(gpa, &storage, &textures);

    std.log.info("uploaded: {d} meshes, {d} texture sets, {d} textures cached", .{
        storage.meshCount(),
        storage.textureSetCount(),
        textures.count(),
    });

    // A batch dropped without finishing, which is the path the transaction exists
    // for. It frees its command buffer before destroying what it registered,
    // because that buffer holds commands naming those resources.
    {
        var rolled_back = try gpu.UploadBatch.begin(
            gpa,
            &context,
            &allocator,
            &storage,
            &textures,
            &staging,
            setup_pool.handle,
        );
        _ = try rolled_back.addMesh(u16, .{
            .vertices = &quad,
            .indices = &quad_indices,
        });
        _ = try rolled_back.addTextureSet(.{});
        std.log.info("rollback: {d} meshes and {d} sets registered", .{
            storage.meshCount(),
            storage.textureSetCount(),
        });
        rolled_back.deinit();
    }
    std.log.info("rollback: {d} meshes and {d} sets after it", .{
        storage.meshCount(),
        storage.textureSetCount(),
    });

    // Trimming returns blocks that no allocation is left in. A block still
    // holding the surviving mesh's buffers stays, so freeing nothing here is the
    // correct outcome rather than a failure.
    const blocks_before = allocator.liveBlockCount();
    allocator.trim();
    std.log.info("memory: {d} blocks, {d} after trim", .{
        blocks_before,
        allocator.liveBlockCount(),
    });

    // Binding is state, valid outside a render pass. Drawing is not, so
    // Mesh.draw stays uncompiled until there is one.
    const binding = try gpu.beginOneShot(&context, setup_pool.handle);
    storage.mesh(uploaded.meshes.items[0]).?.bind(&context, binding);
    try gpu.submitOneShotAndWait(&context, setup_pool.handle, binding);

    // Every submission has completed, so the arena is free again. Nothing here
    // tracks that for the caller.
    staging.reset();
    std.log.info("staging: {d} of {d} bytes free after reset", .{
        staging.remaining(),
        staging.capacity(),
    });

    // Samplers are cached by value, so the same configuration asked for twice is
    // created once.
    var samplers: gpu.SamplerCache = .init(&context);
    defer samplers.deinit(gpa);
    const repeat = try samplers.get(gpa, .{});
    const clamped = try samplers.get(gpa, .{ .address_mode_u = .clamp_to_edge });
    const repeat_again = try samplers.get(gpa, .{});
    std.log.info("samplers: {d} distinct, repeat reused {}", .{
        samplers.count(),
        repeat == repeat_again and repeat != clamped,
    });

    // Materials are packed on the host and uploaded once. The buffer is a single
    // copy shared by every frame, so this is configure-time work, not per-frame.
    var materials: gpu.MaterialStorage = try .init(&context, &allocator, 64);
    defer materials.deinit();

    var info: res.MaterialInfo = .{
        .name = "example",
        .textures = .{},
        .factors = .{ .metallic = 0.0, .roughness = 0.4 },
        .rendering = .{ .alpha_mode = .mask },
    };
    info.textures.base_colour.path = "example.ktx2";
    try materials.upload(&.{gpu.MaterialData.fromInfo(&info)});
    std.log.info("materials: {d} of {d} packed, {d} bytes", .{
        materials.count,
        materials.capacity,
        materials.byteSize(),
    });

    // This example composes and tears down; it does not present. The frame loop
    // belongs to the engine, and a module's example demonstrates the module.
    std.log.info("core composed and torn down", .{});
}
