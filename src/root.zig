const buffer = @import("buffer.zig");
const commands = @import("commands.zig");
const context = @import("context.zig");
const frame = @import("frame.zig");
const image = @import("image.zig");
const ktx2 = @import("ktx2.zig");
const material_storage = @import("material_storage.zig");
const memory = @import("memory/allocator.zig");
const mesh = @import("mesh/resource.zig");
const vertex = @import("mesh/vertex.zig");
const pool = @import("pool.zig");
const ref_cache = @import("ref_cache.zig");
const sampler = @import("sampler.zig");
const resource_storage = @import("resource_storage.zig");
const staging = @import("staging/arena.zig");
const texture_cache = @import("texture_cache.zig");
const upload = @import("upload.zig");
const storage = @import("storage.zig");
const swapchain = @import("swapchain.zig");

pub const Context = context.Context;
pub const Queue = context.Queue;
pub const Frame = frame.Frame;
pub const Swapchain = swapchain.Swapchain;
pub const PresentModePreference = swapchain.PresentModePreference;

pub const MemoryAllocator = memory.MemoryAllocator;
pub const MemoryConfig = memory.Config;
pub const BufferMemoryClass = memory.BufferClass;
pub const MemoryAllocation = memory.Allocation;

pub const Buffer = buffer.Buffer;
pub const CopyRegion = buffer.CopyRegion;
pub const BufferDescriptor = buffer.Descriptor;
pub const BufferCreateRequest = buffer.CreateRequest;
pub const validateBufferCreate = buffer.validateCreate;
pub const validateBufferCopy = buffer.validateCopy;

pub const Image = image.Image;
pub const ImageConfig = image.Config;
pub const MipCopy = image.MipCopy;
pub const LayoutTransition = image.LayoutTransition;

pub const Ktx2File = ktx2.File;
pub const Ktx2Level = ktx2.Level;
pub const parseKtx2 = ktx2.parse;
pub const isKtx2 = ktx2.isKtx2;

pub const MaterialData = material_storage.MaterialData;
pub const MaterialStorage = material_storage.MaterialStorage;
pub const MaterialTextureSlot = material_storage.TextureSlot;

pub const SamplerCache = sampler.SamplerCache;

pub const Mesh = mesh.Mesh;
pub const MeshUpload = mesh.Upload;
pub const MorphUpload = mesh.MorphUpload;
pub const GpuVertex = vertex.GpuVertex;
pub const GpuSkinVertex = vertex.GpuSkinVertex;
pub const GpuColourVertex = vertex.GpuColourVertex;
pub const GpuUv1Vertex = vertex.GpuUv1Vertex;
pub const packVertex = vertex.packVertex;
pub const packSkinVertex = vertex.packSkinVertex;
pub const packColourVertex = vertex.packColourVertex;
pub const packUv1Vertex = vertex.packUv1Vertex;
pub const packDirection = vertex.packDirection;
pub const packSnorm3x10_1x2 = vertex.packSnorm3x10_1x2;

pub const UploadBatch = upload.Batch;
pub const Uploaded = upload.Uploaded;
pub const TextureRequest = upload.TextureRequest;
pub const TextureSetRequest = upload.TextureSetRequest;

pub const ResourceStorage = resource_storage.ResourceStorage;
pub const TextureSet = resource_storage.TextureSet;
pub const MeshHandle = resource_storage.MeshHandle;
pub const TextureSetHandle = resource_storage.TextureSetHandle;

pub const TextureCache = texture_cache.TextureCache;
pub const TextureFallback = texture_cache.Fallback;
pub const BoundTexture = texture_cache.Bound;

pub const StagingArena = staging.StagingArena;
pub const StagingReservation = staging.Reservation;

pub const ResourcePool = pool.ResourcePool;
pub const PoolAddError = pool.AddError;
pub const OwningStorage = storage.OwningStorage;

pub const RefCache = ref_cache.RefCache;
pub const RefCacheInsertError = ref_cache.InsertError;
pub const RefCacheDeinitStatus = ref_cache.DeinitStatus;

pub const OneShotPool = commands.OneShotPool;
pub const beginOneShot = commands.beginOneShot;
pub const submitOneShotAndWait = commands.submitOneShotAndWait;
