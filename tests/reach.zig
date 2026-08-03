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

    _ = &gpu.StagingArena.init;
    _ = &gpu.StagingArena.deinit;
    _ = &gpu.StagingArena.reserve;
    _ = &gpu.StagingArena.reset;
    _ = &gpu.StagingArena.source;

    _ = &gpu.TextureCache.init;
    _ = &gpu.TextureCache.deinit;
    _ = &gpu.TextureCache.count;

    _ = &gpu.SamplerCache.init;
    _ = &gpu.SamplerCache.deinit;
    _ = &gpu.SamplerCache.get;
    _ = &gpu.SamplerCache.count;

    _ = &gpu.MaterialStorage.init;
    _ = &gpu.MaterialStorage.deinit;
    _ = &gpu.MaterialStorage.upload;
    _ = &gpu.MaterialStorage.byteSize;

    _ = &gpu.Mesh.init;
    _ = &gpu.Mesh.deinit;
    _ = &gpu.Mesh.bind;
    _ = &gpu.Mesh.draw;

    _ = &gpu.UploadBatch.begin;
    _ = &gpu.UploadBatch.deinit;
    _ = &gpu.UploadBatch.addMesh;
    _ = &gpu.UploadBatch.addTextureSet;
    _ = &gpu.UploadBatch.finish;

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
    _ = &gpu.Renderer.record;
    _ = &gpu.Renderer.bindMaterial;
    _ = &gpu.Renderer.targetExtent;
    _ = &gpu.Renderer.mainPassTarget;

    _ = &gpu.Attachment.createHdr;
    _ = &gpu.Attachment.createDepth;

    _ = &gpu.Swapchain.currentExtent;
    _ = &gpu.Swapchain.matchesExtent;
    _ = &gpu.Swapchain.renderFinishedSemaphore;
    _ = &gpu.TextureCache.acquireKtx2;
    _ = &gpu.ResourceStorage.removeMesh;
    _ = &gpu.ResourceStorage.removeTextureSet;
    _ = &gpu.MemoryAllocation.mappedBytes;
    _ = &gpu.LayoutTransition.toShaderRead;
    _ = &gpu.mipExtent;
}
