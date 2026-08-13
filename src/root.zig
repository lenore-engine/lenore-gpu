const attachment = @import("attachment.zig");
const buffer = @import("buffer.zig");
const commands = @import("commands.zig");
const context = @import("context.zig");
const descriptors = @import("descriptors.zig");
const effect = @import("effect.zig");
const frame = @import("frame.zig");
const frame_set = @import("frame_set.zig");
const image = @import("image.zig");
const ktx2 = @import("ktx2.zig");
const material_storage = @import("material_storage.zig");
const memory = @import("memory/allocator.zig");
const mesh = @import("mesh/resource.zig");
const morph = @import("morph.zig");
const vertex = @import("mesh/vertex.zig");
const pass = @import("pass.zig");
const pipeline = @import("pipeline.zig");
const timing = @import("timing.zig");
const per_frame = @import("per_frame.zig");
const post = @import("post.zig");
const pool = @import("pool.zig");
const ref_cache = @import("ref_cache.zig");
const retirement = @import("retirement.zig");
const sampler = @import("sampler.zig");
const renderer = @import("renderer.zig");
const resource_storage = @import("resource_storage.zig");
const staging = @import("staging/pool.zig");
const environment = @import("environment.zig");
const texture_cache = @import("texture_cache.zig");
const transfer = @import("transfer.zig");
const uniforms = @import("uniforms.zig");
const upload = @import("upload.zig");
const storage = @import("storage.zig");
const swapchain = @import("swapchain.zig");

pub const Context = context.Context;
pub const Queue = context.Queue;
pub const validationErrorCount = context.validationErrorCount;
pub const QueueAllocation = context.QueueAllocation;
pub const QueueSupport = context.QueueSupport;
pub const chooseQueues = context.chooseQueues;
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
pub const ImageShape = image.Shape;
pub const imageFormatSupports = image.formatSupports;
pub const imageUsageSupported = image.usageSupported;
pub const LayoutTransition = image.LayoutTransition;
pub const mipExtent = image.mipExtent;

pub const Ktx2File = ktx2.File;
pub const Ktx2Kind = ktx2.Kind;
pub const Ktx2Level = ktx2.Level;
pub const parseKtx2 = ktx2.parse;
pub const ktx2CubeFaces = ktx2.cube_faces;
pub const isKtx2 = ktx2.isKtx2;

pub const MaterialData = material_storage.MaterialData;
pub const MaterialStorage = material_storage.MaterialStorage;
pub const MaterialTextureSlot = material_storage.TextureSlot;
pub const MaterialArrayBindings = material_storage.bindings;
pub const Shading = @import("shading.zig");

pub const SamplerCache = sampler.SamplerCache;
pub const samplerCreateInfo = sampler.createInfo;

pub const Mesh = mesh.Mesh;
pub const MeshUpload = mesh.Upload;
pub const MorphUpload = mesh.MorphUpload;
pub const MeshVertexSource = mesh.VertexSource;
pub const meshVertexUsage = mesh.vertexUsage;

pub const MorphPass = morph.MorphPass;
pub const MorphCapacity = morph.Capacity;
pub const MorphPushConstants = morph.PushConstants;
pub const morph_bindings = morph.bindings;
pub const morphDependency = morph.dependency;
pub const morphGroupSize = morph.group_size;
pub const morphGroupsFor = morph.groupsFor;
pub const morphDestinationElements = morph.destinationElements;
pub const morphReserveWeights = morph.reserveWeights;
pub const morphValidateRegistration = morph.validateRegistration;
pub const morphValidateWeightCount = morph.validateWeightCount;
pub const morphPushConstantRange = morph.push_constant_range;
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
pub const SunShadowUniform = uniforms.SunShadow;
pub const max_lights = uniforms.max_lights;
pub const vulkanClip = uniforms.vulkanClip;
pub const vulkanClipCamera = uniforms.vulkanClipCamera;
pub const FramebufferCamera = uniforms.FramebufferCamera;
pub const Uniforms = uniforms;

const shadow = @import("shadow.zig");
pub const ShadowPass = shadow.ShadowPass;
pub const Shadow = shadow;

const sky = @import("sky.zig");
pub const Sky = sky;
pub const Background = sky.Background;

const bloom = @import("bloom.zig");
pub const Bloom = bloom;
pub const BloomPass = bloom.BloomPass;
pub const BloomSettings = bloom.Settings;
pub const BloomSettingsError = bloom.SettingsError;
pub const BloomLook = bloom.Look;
pub const BloomPushConstants = bloom.PushConstants;
pub const BloomLevelTransition = bloom.LevelTransition;
pub const bloom_bindings = bloom.bindings;
pub const bloom_max_levels = bloom.max_levels;
pub const bloomResolve = bloom.resolve;
pub const bloomBaseExtent = bloom.baseExtent;
pub const bloomChainDepth = bloom.chainDepth;
pub const bloomTexelSize = bloom.texelSize;
pub const bloomLevelBarrier = bloom.levelBarrierFor;
pub const bloomPushConstantRange = bloom.push_constant_range;

pub const UploadBatch = upload.Batch;
pub const Uploaded = upload.Uploaded;
pub const TextureRequest = upload.TextureRequest;
pub const TextureSlot = upload.TextureSlot;
pub const TextureSetRequest = upload.TextureSetRequest;
pub const MaterialSlot = upload.MaterialSlot;

pub const Renderer = renderer.Renderer;
pub const RendererMaterialError = renderer.MaterialError;
pub const RendererFrameError = renderer.FrameError;
pub const RendererUpdateError = renderer.UpdateError;
pub const RendererRecordError = renderer.RecordError;
pub const validateRendererMaterialIndex = renderer.validateMaterialIndex;
pub const validateRendererFrameIndex = renderer.validateFrameIndex;
pub const validateRendererRecordBatch = renderer.validateRecordBatch;
pub const backgroundSlot = renderer.backgroundSlot;
pub const RecordRequest = renderer.RecordRequest;
pub const RecordState = renderer.RecordState;
pub const RecordPlan = renderer.RecordPlan;
pub const planRecording = renderer.planRecording;
pub const SceneVariant = renderer.SceneVariant;
pub const sceneVariantFor = renderer.sceneVariantFor;
pub const sceneVariantIndex = renderer.sceneVariantIndex;
pub const scenePipelineIndex = renderer.scenePipelineIndex;
pub const scene_variants = renderer.scene_variants;
pub const scene_pipeline_count = renderer.scene_pipeline_count;
pub const scene_modes = renderer.scene_modes;
pub const RecordBatch = renderer.RecordBatch;
pub const MaterialRecord = renderer.MaterialRecord;
pub const ShadowBake = renderer.ShadowBake;
pub const batchVertexSource = renderer.batchVertexSource;
// What the shading this module is built from has to supply, one struct per
// pass. The words are authored by whoever owns the look; nothing here embeds
// any, and these are the shapes that composition fills.
pub const Shaders = renderer.Shaders;
pub const SceneShader = renderer.SceneShader;
pub const SkyShader = sky.Shader;
pub const PostShader = post.Shader;
pub const BloomShader = bloom.Shader;
pub const ShadowShader = shadow.Shader;
pub const MorphShader = morph.Shader;

pub const RendererMaterialBindings = renderer.material_bindings;
pub const SceneSetBindings = renderer.scene_bindings;
pub const frame_set_index = renderer.frame_set_index;
pub const scene_set_index = renderer.scene_set_index;
pub const material_set_index = renderer.material_set_index;
pub const shadow_set_index = renderer.shadow_set_index;

pub const ResourceStorage = resource_storage.ResourceStorage;
pub const TextureSet = resource_storage.TextureSet;
pub const MeshHandle = resource_storage.MeshHandle;
pub const TextureSetHandle = resource_storage.TextureSetHandle;

pub const TextureCache = texture_cache.TextureCache;
pub const TextureFallback = texture_cache.Fallback;
pub const BoundTexture = texture_cache.Bound;
pub const ResidentTexture = texture_cache.Resident;
pub const ktx2ImageShape = texture_cache.imageShape;
pub const Environment = environment.Environment;
pub const environment_bindings = environment.bindings;
pub const environmentSampler = environment.sampler_config;
pub const writeEnvironment = environment.write;

pub const StagingPool = staging.StagingPool;
pub const StagingConfig = staging.Config;
pub const StagingReservation = staging.Reservation;

pub const Transfer = transfer.Transfer;

pub const Attachment = attachment;
pub const attachmentDepthFormat = attachment.depthFormat;
pub const attachmentHdrFormat = attachment.hdrFormat;
pub const attachmentFirstSupported = attachment.firstSupported;

pub const DescriptorBinding = descriptors.Binding;
pub const DescriptorSets = descriptors.Sets;
pub const DescriptorBufferSource = descriptors.BufferSource;
pub const DescriptorImageSource = descriptors.ImageSource;

// The bindings this module was generated against.
//
// A caller recording draws of its own speaks Vulkan; there is no way around
// that, and a wrapper over `Pipeline` above would be a second way to do what is
// already public. What the caller must not do is generate its own bindings: two
// generations from the same registry produce two distinct sets of types, so a
// pipeline built by one would not fit a command buffer named by the other, and
// the compiler would say so somewhere far from the cause.
//
// One generation, reached through the module that owns it.
pub const vk = @import("vulkan");
pub const DescriptorType = vk.DescriptorType;

pub const MainPass = pass;
pub const MainPassTarget = pass.Target;
pub const MainPassOptions = pass.Options;
pub const mainPassSampledLayout = pass.sampled_layout;

pub const Pipeline = pipeline;
pub const ShaderEffect = effect.ShaderEffect;
pub const ShaderEffectSpec = effect.Spec;
pub const ShaderEffectStage = effect.Stage;
pub const ShaderEffectGraphics = effect.Graphics;
pub const PipelineConfig = pipeline.Config;
pub const PipelineLayoutConfig = pipeline.LayoutConfig;
pub const PipelineMode = pipeline.Mode;
pub const PipelineFormats = pipeline.Formats;
pub const PipelineStages = pipeline.Stages;
pub const PipelineStage = pipeline.Stage;
pub const pipelineVertexInput = pipeline.vertexInput;
pub const pipelineModeFor = pipeline.modeFor;
pub const pipelineDepthStencilState = pipeline.depthStencilState;
pub const pipelineBlendAttachment = pipeline.blendAttachment;
pub const captureShaderStatistics = pipeline.capture_statistics;
pub const PipelineStatisticsError = pipeline.StatisticsError;
pub const pipelineExecutables = pipeline.executables;
pub const pipelineStatistics = pipeline.statistics;
pub const pipelineDescribedName = pipeline.describedName;

pub const GpuTimer = timing.GpuTimer;
pub const GpuTimerSupport = timing.Support;
pub const GpuTimings = timing.Frame;
pub const GpuPass = timing.Pass;
pub const GpuTimestampEdge = timing.Edge;
pub const gpuTimestampSlot = timing.slot;
pub const gpuDurationNs = timing.durationNs;

pub const PerFrame = per_frame.PerFrame;
pub const perFrameLayout = per_frame.layout;
pub const perFrameOffsetAlignment = per_frame.offsetAlignment;

pub const PostPass = post;
pub const PostTarget = post.Target;
pub const PostSets = post.Sets;
pub const PostSource = post.Source;
pub const PostBindings = post.bindings;
pub const PostSettings = post.Settings;
pub const PostSettingsError = post.SettingsError;
pub const PostPushConstants = post.PushConstants;
pub const postPushConstantRange = post.push_constant_range;
pub const toneMap = post.toneMap;

pub const ResourcePool = pool.ResourcePool;
pub const PoolAddError = pool.AddError;
pub const OwningStorage = storage.OwningStorage;

pub const RefCache = ref_cache.RefCache;
pub const RefCacheInsertError = ref_cache.InsertError;
pub const RefCacheDeinitStatus = ref_cache.DeinitStatus;

pub const Retirement = retirement.Retirement;
pub const RetirementInitError = retirement.InitError;
pub const RetiredResource = retirement.Resource;
pub const ResourceRetirement = retirement.ResourceRetirement;
pub const retireOrDestroy = retirement.retireOrDestroy;

pub const OneShotPool = commands.OneShotPool;
pub const beginOneShot = commands.beginOneShot;
pub const submitOneShotAndWait = commands.submitOneShotAndWait;
pub const Dependency = commands.Dependency;
pub const memoryBarrier = commands.memoryBarrier;
pub const recordMemoryBarrier = commands.recordMemoryBarrier;
