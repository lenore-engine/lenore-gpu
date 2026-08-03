const attachment = @import("attachment.zig");
const buffer = @import("buffer.zig");
const commands = @import("commands.zig");
const context = @import("context.zig");
const descriptors = @import("descriptors.zig");
const frame = @import("frame.zig");
const frame_set = @import("frame_set.zig");
const image = @import("image.zig");
const ktx2 = @import("ktx2.zig");
const material_storage = @import("material_storage.zig");
const memory = @import("memory/allocator.zig");
const mesh = @import("mesh/resource.zig");
const vertex = @import("mesh/vertex.zig");
const pass = @import("pass.zig");
const pipeline = @import("pipeline.zig");
const per_frame = @import("per_frame.zig");
const post = @import("post.zig");
const pool = @import("pool.zig");
const ref_cache = @import("ref_cache.zig");
const sampler = @import("sampler.zig");
const renderer = @import("renderer.zig");
const resource_storage = @import("resource_storage.zig");
const shaders = @import("shaders.zig");
const staging = @import("staging/arena.zig");
const texture_cache = @import("texture_cache.zig");
const uniforms = @import("uniforms.zig");
const upload = @import("upload.zig");
const storage = @import("storage.zig");
const swapchain = @import("swapchain.zig");

pub const Context = context.Context;
pub const Queue = context.Queue;
pub const validationErrorCount = context.validationErrorCount;
pub const Frame = frame.Frame;
pub const FrameSet = frame_set.FrameSet;
pub const FrameSetBindings = frame_set.bindings;
pub const FrameContents = frame_set.FrameSet.Frame;
pub const validateFrameContents = frame_set.validate;
pub const Instance = frame_set.Instance;
pub const Joint = frame_set.Joint;
pub const FrameCapacity = frame_set.Capacity;
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
pub const ImageKind = image.Kind;
pub const imageFormatSupports = image.formatSupports;
pub const imageUsageSupported = image.usageSupported;
pub const LayoutTransition = image.LayoutTransition;
pub const mipExtent = image.mipExtent;

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

pub const CameraUniform = uniforms.Camera;
pub const LightUniform = uniforms.Light;
pub const LightsUniform = uniforms.Lights;
pub const max_lights = uniforms.max_lights;
pub const vulkanClip = uniforms.vulkanClip;
pub const Uniforms = uniforms;

pub const UploadBatch = upload.Batch;
pub const Uploaded = upload.Uploaded;
pub const TextureRequest = upload.TextureRequest;
pub const TextureSetRequest = upload.TextureSetRequest;

pub const Renderer = renderer.Renderer;
pub const SceneVariant = renderer.SceneVariant;
pub const sceneVariantFor = renderer.sceneVariantFor;
pub const RendererDraw = renderer.Draw;
pub const RendererMaterialBindings = renderer.material_bindings;

pub const ResourceStorage = resource_storage.ResourceStorage;
pub const TextureSet = resource_storage.TextureSet;
pub const MeshHandle = resource_storage.MeshHandle;
pub const TextureSetHandle = resource_storage.TextureSetHandle;

pub const TextureCache = texture_cache.TextureCache;
pub const TextureFallback = texture_cache.Fallback;
pub const BoundTexture = texture_cache.Bound;

pub const Shaders = shaders;
pub const ShaderModule = shaders.Module;
pub const ShaderEntryPoint = shaders.EntryPoint;

pub const StagingArena = staging.StagingArena;
pub const StagingReservation = staging.Reservation;

pub const Attachment = attachment;
pub const attachmentDepthFormat = attachment.depthFormat;
pub const attachmentHdrFormat = attachment.hdrFormat;
pub const attachmentFirstSupported = attachment.firstSupported;

pub const DescriptorBinding = descriptors.Binding;
pub const DescriptorSets = descriptors.Sets;

pub const MainPass = pass;
pub const MainPassTarget = pass.Target;
pub const MainPassOptions = pass.Options;
pub const mainPassSampledLayout = pass.sampled_layout;

pub const Pipeline = pipeline;
pub const PipelineConfig = pipeline.Config;
pub const PipelineMode = pipeline.Mode;
pub const PipelineFormats = pipeline.Formats;
pub const PipelineStages = pipeline.Stages;
pub const PipelineStage = pipeline.Stage;
pub const pipelineVertexInput = pipeline.vertexInput;
pub const pipelineModeFor = pipeline.modeFor;
pub const pipelineDepthStencilState = pipeline.depthStencilState;
pub const pipelineBlendAttachment = pipeline.blendAttachment;

pub const PerFrame = per_frame.PerFrame;
pub const perFrameLayout = per_frame.layout;
pub const perFrameOffsetAlignment = per_frame.offsetAlignment;

pub const PostPass = post;
pub const PostTarget = post.Target;
pub const PostSets = post.Sets;
pub const PostSource = post.Source;
pub const PostBindings = post.bindings;

pub const ResourcePool = pool.ResourcePool;
pub const PoolAddError = pool.AddError;
pub const OwningStorage = storage.OwningStorage;

pub const RefCache = ref_cache.RefCache;
pub const RefCacheInsertError = ref_cache.InsertError;
pub const RefCacheDeinitStatus = ref_cache.DeinitStatus;

pub const OneShotPool = commands.OneShotPool;
pub const beginOneShot = commands.beginOneShot;
pub const submitOneShotAndWait = commands.submitOneShotAndWait;
