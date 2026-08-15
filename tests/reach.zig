const std = @import("std");
const gpu = @import("lenore-gpu");

// What the compiler would otherwise never look at.
//
// A test reaches only what it calls, and most of this module's surface needs a
// device to call. Referencing a function compiles its body, so this is what
// makes `zig build test` from this directory a real check on the whole module
// rather than on the arithmetic that happens to be host-side.
//
// Proven by breaking a body and watching this fail: without it, `Mesh.bind`
// could be made nonsense and the test step stayed green.
//
// Anything here that gains a test which actually calls it should lose its line.

test "the device-facing surface is compiled" {
    _ = &gpu.GpuTimer.init;
    _ = &gpu.GpuTimer.deinit;
    _ = &gpu.GpuTimer.reset;
    _ = &gpu.GpuTimer.write;
    _ = &gpu.GpuTimer.read;

    // Both need a device and a pipeline created with the capture flag, so
    // nothing here can call them.
    _ = &gpu.pipelineExecutables;
    _ = &gpu.pipelineStatistics;

    _ = &gpu.Context.init;
    _ = &gpu.Context.deinit;
    _ = &gpu.Context.deviceName;
    _ = &gpu.Context.waitIdle;
    _ = &gpu.validationErrorCount;

    _ = &gpu.Swapchain.init;
    _ = &gpu.Swapchain.deinit;
    _ = &gpu.Swapchain.recreate;
    _ = &gpu.Swapchain.acquireNextImage;
    _ = &gpu.Swapchain.present;
    _ = &gpu.Swapchain.recordClear;

    _ = &gpu.Frame.init;
    _ = &gpu.Frame.deinit;
    _ = &gpu.Frame.waitForGpu;
    _ = &gpu.Frame.beginCommands;
    _ = &gpu.Frame.submit;

    _ = &gpu.MemoryAllocator.init;
    _ = &gpu.MemoryAllocator.deinit;
    _ = &gpu.MemoryAllocator.allocateBuffer;
    _ = &gpu.MemoryAllocator.allocateOptimalImage;
    _ = &gpu.MemoryAllocator.free;
    _ = &gpu.MemoryAllocator.trim;
    _ = &gpu.MemoryAllocator.liveBlockCount;

    _ = &gpu.Buffer.init;
    _ = &gpu.Buffer.deinit;
    _ = &gpu.Buffer.upload;
    _ = &gpu.Buffer.uploadAt;
    _ = &gpu.Buffer.mapped;
    _ = &gpu.Buffer.describe;
    _ = &gpu.Buffer.recordCopyFrom;

    _ = &gpu.Image.init;
    _ = &gpu.Image.deinit;
    _ = &gpu.Image.recordLayoutTransition;
    _ = &gpu.Image.recordCopyFrom;

    _ = &gpu.StagingPool.init;
    _ = &gpu.StagingPool.deinit;
    _ = &gpu.StagingPool.reserve;
    _ = &gpu.StagingPool.recycle;
    _ = &gpu.StagingPool.blockCount;
    _ = &gpu.StagingPool.residentBytes;
    _ = &gpu.StagingPool.blockCapacity;

    _ = &gpu.Transfer.begin;
    _ = &gpu.Transfer.deinit;
    _ = &gpu.Transfer.reserve;
    _ = &gpu.Transfer.flush;
    _ = &gpu.Transfer.finish;
    _ = &gpu.Transfer.commandBuffer;
    _ = &gpu.Transfer.abandoned;

    _ = &gpu.TextureCache.init;
    _ = &gpu.TextureCache.deinit;
    _ = &gpu.TextureCache.count;
    // Releasing routes an image into the frame ring and falls back to a drain,
    // both of which are device calls. Nothing else in this module's own suite
    // compiles either path.
    _ = &gpu.TextureCache.release;
    _ = &gpu.TextureCache.pin;

    // `Retirement` itself is tested against a value that needs no device, so
    // what is left here is the concrete union: every variant destroys through a
    // device, and the dispatch is compiled nowhere else in this suite. A variant
    // added without a `deinit` fails at this line.
    _ = &gpu.RetiredResource.deinit;
    _ = &gpu.ResourceRetirement.init;
    _ = &gpu.ResourceRetirement.deinit;
    _ = &gpu.ResourceRetirement.beginFrame;
    _ = &gpu.ResourceRetirement.retire;
    // The degradation the two release paths share. Its fallback drains a device,
    // so nothing host-side reaches it.
    _ = &gpu.retireOrDestroy;

    _ = &gpu.SamplerCache.init;
    _ = &gpu.SamplerCache.deinit;
    _ = &gpu.SamplerCache.get;
    _ = &gpu.SamplerCache.count;

    _ = &gpu.MaterialStorage.init;
    _ = &gpu.MaterialStorage.deinit;
    _ = &gpu.MaterialStorage.upload;
    _ = &gpu.MaterialStorage.byteSize;
    _ = &gpu.MaterialStorage.descriptor;

    _ = &gpu.Mesh.deinit;
    _ = &gpu.Mesh.bind;
    _ = &gpu.Mesh.draw;

    _ = &gpu.Renderer.beginPost;
    _ = &gpu.Renderer.endPost;

    _ = &gpu.UiPass.init;
    _ = &gpu.UiPass.deinit;
    _ = &gpu.UiPass.storage;
    _ = &gpu.UiPass.record;
    _ = &gpu.UiRegistry.init;
    _ = &gpu.UiRegistry.deinit;
    _ = &gpu.UiRegistry.add;
    _ = &gpu.UiRegistry.remove;
    _ = &gpu.UiRegistry.descriptorSet;
    _ = &gpu.UiRegistry.count;

    _ = &gpu.MorphPass.init;
    _ = &gpu.MorphPass.deinit;
    _ = &gpu.MorphPass.register;
    _ = &gpu.MorphPass.registrationCount;
    _ = &gpu.MorphPass.reset;
    _ = &gpu.MorphPass.writeWeights;
    _ = &gpu.MorphPass.vertexSource;
    _ = &gpu.MorphPass.record;
    _ = &gpu.MorphPass.descriptorSetLayout;

    _ = &gpu.ShadowPass.init;
    _ = &gpu.ShadowPass.deinit;
    _ = &gpu.ShadowPass.descriptorSetLayout;
    _ = &gpu.ShadowPass.descriptorSet;
    _ = &gpu.ShadowPass.mapSize;
    _ = &gpu.ShadowPass.pipelineFor;
    _ = &gpu.ShadowPass.begin;
    _ = &gpu.ShadowPass.end;

    _ = &gpu.Sky.record;

    _ = &gpu.BloomPass.init;
    _ = &gpu.BloomPass.deinit;
    _ = &gpu.BloomPass.recreate;
    _ = &gpu.BloomPass.record;
    _ = &gpu.BloomPass.levelCount;
    _ = &gpu.BloomPass.compositeView;
    _ = &gpu.BloomPass.compositeSampler;

    _ = &gpu.UploadBatch.begin;
    _ = &gpu.UploadBatch.deinit;
    _ = &gpu.UploadBatch.addTextureSet;
    _ = &gpu.UploadBatch.finish;

    // The generic entry points, through the callers below. See why there.
    _ = &reachMeshInit;
    _ = &reachAddMesh;
    _ = &reachAddTexture;

    _ = &gpu.ResourceStorage.deinit;
    _ = &gpu.ResourceStorage.mesh;
    _ = &gpu.ResourceStorage.textureSet;
    _ = &gpu.ResourceStorage.meshCount;
    _ = &gpu.ResourceStorage.textureSetCount;

    _ = &gpu.OneShotPool.init;
    _ = &gpu.OneShotPool.deinit;
    _ = &gpu.beginOneShot;
    _ = &gpu.submitOneShotAndWait;

    _ = &gpu.Renderer.init;
    _ = &gpu.Renderer.deinit;
    _ = &gpu.Renderer.resize;
    _ = &gpu.Renderer.update;
    _ = &gpu.Renderer.plan;
    // The frame's stages, in the order they are recorded. Named one by one and
    // not through a whole-frame entry point, because there is no longer one:
    // composition sequences them.
    _ = &gpu.Renderer.recordShadowBake;
    _ = &gpu.Renderer.beginMain;
    _ = &gpu.Renderer.recordScene;
    _ = &gpu.Renderer.endMain;
    _ = &gpu.Renderer.recordBloom;
    _ = &gpu.Renderer.recordPost;
    _ = &gpu.Renderer.setMaterialBuffer;
    _ = &gpu.Renderer.setEnvironment;
    _ = &gpu.Renderer.setMaterialTextures;
    _ = &gpu.Renderer.clearMaterials;
    _ = &gpu.Renderer.shadowBakes;
    _ = &gpu.Renderer.backgroundDraws;
    _ = &gpu.Renderer.shadowMapSize;
    _ = &gpu.Renderer.targetExtent;
    _ = &gpu.Renderer.mainPassTarget;
    _ = &gpu.Renderer.mainPassFormats;

    _ = &gpu.Attachment.createHdr;
    _ = &gpu.Attachment.createDepth;

    _ = &gpu.Swapchain.currentExtent;
    _ = &gpu.Swapchain.matchesExtent;
    _ = &gpu.Swapchain.renderFinishedSemaphore;
    _ = &gpu.TextureCache.acquireKtx2;
    _ = &gpu.Environment.neutral;
    _ = &gpu.writeEnvironment;
    _ = &gpu.ResourceStorage.removeMesh;
    _ = &gpu.ResourceStorage.removeTextureSet;
    _ = &gpu.MemoryAllocation.mappedBytes;
    _ = &gpu.LayoutTransition.toShaderRead;
    _ = &gpu.mipExtent;
}

// Taking the address of a generic function does not instantiate it, so a line
// for one in the list above compiles nothing. Proven the same way as the rest of
// this file: with `addMesh` listed there and nothing else calling it, its body
// was made nonsense and `zig build test` stayed green.
//
// A caller fixes the comptime arguments, and compiling the caller is what
// compiles the instantiation. One instantiation per function is enough for what
// this file checks, which is that the body is analysed at all.

fn reachMeshInit(
    context: *const gpu.Context,
    memory_allocator: *gpu.MemoryAllocator,
    transfer: *gpu.Transfer,
    upload: gpu.MeshUpload(u32),
) !gpu.Mesh {
    return gpu.Mesh.init(u32, context, memory_allocator, transfer, upload);
}

fn reachAddMesh(batch: *gpu.UploadBatch, upload: gpu.MeshUpload(u32)) !gpu.MeshHandle {
    return batch.addMesh(u32, upload);
}

fn reachAddTexture(batch: *gpu.UploadBatch, request: gpu.TextureRequest) !gpu.BoundTexture {
    return batch.addTexture(.base_colour, request);
}

// The pipeline table. Every method creates or destroys device objects, and the
// examples that use it are built from the umbrella rather than from here.
test "the shader effect table is compiled" {
    const Table = gpu.ShaderEffect(.{
        .modules = .{"only"},
        .layouts = .{"only"},
        .pipelines = .{
            .step = gpu.ShaderEffectSpec{ .module = "only", .layout = "only", .stage = .{
                .compute = "computeMain",
            } },
            .present = gpu.ShaderEffectSpec{ .module = "only", .layout = "only", .stage = .{ .graphics = .{
                .vertex = "vertexMain",
                .fragment = "fragmentMain",
                .mode = .background,
                .culling = .{ .fixed = .{} },
            } } },
        },
    });
    _ = &Table.init;
    _ = &Table.deinit;
    _ = &Table.get;
    _ = &Table.layoutFor;
}
