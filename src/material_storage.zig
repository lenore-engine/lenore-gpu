const std = @import("std");
const vk = @import("vulkan");
const buffer_module = @import("buffer.zig");
const Context = @import("context.zig").Context;
const descriptors = @import("descriptors.zig");
const MaterialInfo = @import("lenore-resources").MaterialInfo;
const memory = @import("memory/allocator.zig");

const Buffer = buffer_module.Buffer;
const UvTransform = MaterialInfo.TextureMaps.UvTransform;

// The GPU-side material layout and the buffer holding it. Every number here is
// shared with a shader, so the layout is pinned by comptime asserts rather than
// described in a comment: an assert stops being true loudly.

// Slot order in the packed array. A shader indexes it by the same constants, so
// the order is part of the contract and not an implementation detail.
pub const TextureSlot = enum(u32) {
    base_colour = 0,
    metallic_roughness = 1,
    normal = 2,
    emissive = 3,
    occlusion = 4,
};

// Bits of the texture-presence mask. Base colour has no bit: a material without
// one is bound to the neutral white fallback, which makes sampling it a no-op
// against the factor. The other four change what the shader computes, so it has
// to know whether they are real.
pub const texture_present = struct {
    pub const normal: u32 = 1 << 0;
    pub const metallic_roughness: u32 = 1 << 1;
    pub const emissive: u32 = 1 << 2;
    pub const occlusion: u32 = 1 << 3;
};

// One slot's KHR_texture_transform, folded for the shader: rs is the row-major
// 2x2 of rotation times scale, params is offset.xy, the UV set as a float, and
// one unused lane. The shader samples at rs * uv + offset on the selected set.
pub const TexTransform = extern struct {
    rs: [4]f32 align(16),
    params: [4]f32 align(16),

    // The combined glTF transform is translation, then rotation, then scale, so
    // the upper-left 2x2 is rotation times scale and the offset is the raw
    // translation.
    //
    // The sine signs are transposed relative to the column-major GLSL snippet in
    // the extension's own README. UV space has V pointing down, so the
    // extension's positive rotation is clockwise, and the sampling matrix needs
    // the opposite sign. The Khronos TextureTransformTest asset shows which
    // direction is in effect: its rotated quad hits a green marker for the
    // correct one and a red marker for the reverse.
    pub fn fromUv(uv: UvTransform) TexTransform {
        const cosine = @cos(uv.rotation);
        const sine = @sin(uv.rotation);
        return .{
            .rs = .{
                cosine * uv.scale[0], sine * uv.scale[1],
                -sine * uv.scale[0],  cosine * uv.scale[1],
            },
            .params = .{
                uv.offset[0],
                uv.offset[1],
                @floatFromInt(uv.set),
                0.0,
            },
        };
    }
};

pub const MaterialData = extern struct {
    base_colour_factor: [4]f32 align(16),
    // Emissive in xyz, occlusion strength in w.
    emissive_factor: [4]f32 align(16),
    // Metallic, roughness, alpha cutoff, normal scale.
    metallic_roughness_cutoff: [4]f32 align(16),
    // Alpha mode, double sided, the texture-presence mask, unlit.
    flags: [4]u32 align(16),
    tex: [@typeInfo(TextureSlot).@"enum".fields.len]TexTransform align(16),

    comptime {
        // Measured by the compiler, not derived by hand. They pin this side of a
        // layout that a shader has to agree with, and nothing here can check the
        // other side: when the shader exists, a mismatch is a material read at
        // the wrong offsets rather than a crash.
        std.debug.assert(@sizeOf(TexTransform) == 32);
        std.debug.assert(@sizeOf(MaterialData) == 224);
        std.debug.assert(@alignOf(MaterialData) == 16);
        std.debug.assert(@offsetOf(MaterialData, "base_colour_factor") == 0);
        std.debug.assert(@offsetOf(MaterialData, "emissive_factor") == 16);
        std.debug.assert(@offsetOf(MaterialData, "metallic_roughness_cutoff") == 32);
        std.debug.assert(@offsetOf(MaterialData, "flags") == 48);
        std.debug.assert(@offsetOf(MaterialData, "tex") == 64);

        // The alpha mode is packed as its ordinal, so the shader's constants are
        // these values. Renaming or reordering the enum changes the wire format.
        std.debug.assert(@intFromEnum(MaterialInfo.Rendering.AlphaMode.@"opaque") == 0);
        std.debug.assert(@intFromEnum(MaterialInfo.Rendering.AlphaMode.mask) == 1);
        std.debug.assert(@intFromEnum(MaterialInfo.Rendering.AlphaMode.blend) == 2);

        // The packed slot array is indexed by TextureSlot, so its order and the
        // order the fields are written in fromInfo must agree.
        std.debug.assert(@intFromEnum(TextureSlot.base_colour) == 0);
        std.debug.assert(@intFromEnum(TextureSlot.occlusion) == 4);
    }

    // The first flag lane, back as the enum it was packed from. The asserts
    // above pin the three ordinals, and `fromInfo` is the only writer of the
    // lane, so every value in a buffer this module filled names a member.
    pub fn alphaMode(self: MaterialData) MaterialInfo.Rendering.AlphaMode {
        return @enumFromInt(self.flags[0]);
    }

    pub fn fromInfo(info: *const MaterialInfo) MaterialData {
        const textures = &info.textures;
        var mask: u32 = 0;
        if (textures.normal.path != null) mask |= texture_present.normal;
        if (textures.metallic_roughness.path != null) mask |= texture_present.metallic_roughness;
        if (textures.emissive.path != null) mask |= texture_present.emissive;
        if (textures.occlusion.path != null) mask |= texture_present.occlusion;

        return .{
            .base_colour_factor = info.factors.base_colour,
            .emissive_factor = .{
                info.factors.emissive[0],
                info.factors.emissive[1],
                info.factors.emissive[2],
                info.factors.occlusion_strength,
            },
            .metallic_roughness_cutoff = .{
                info.factors.metallic,
                info.factors.roughness,
                info.rendering.alpha_cutoff,
                info.factors.normal_scale,
            },
            .flags = .{
                @intFromEnum(info.rendering.alpha_mode),
                @intFromBool(info.rendering.double_sided),
                mask,
                @intFromBool(info.rendering.unlit),
            },
            .tex = .{
                TexTransform.fromUv(textures.base_colour.uv),
                TexTransform.fromUv(textures.metallic_roughness.uv),
                TexTransform.fromUv(textures.normal.uv),
                TexTransform.fromUv(textures.emissive.uv),
                TexTransform.fromUv(textures.occlusion.uv),
            },
        };
    }
};

pub const InitError = error{ZeroCapacity} || buffer_module.InitError;
pub const UploadError = error{CapacityExceeded} || buffer_module.UploadError;

// One storage buffer of packed materials, host-visible and persistently mapped.
//
// It is a single copy shared by every frame in flight, unlike per-frame data
// which has one copy per slot written behind that slot's fence. A frame may
// therefore be reading it while the host wants to write, so uploading is a cold
// path: the caller waits for the device to go idle first. Writing it per frame
// requires per-slot copies, which do not exist here.
pub const MaterialStorage = struct {
    buffer: Buffer,
    capacity: u32,
    count: u32,

    pub fn init(
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        capacity: u32,
    ) InitError!MaterialStorage {
        if (capacity == 0) return error.ZeroCapacity;
        const buffer = try Buffer.init(
            context,
            memory_allocator,
            @as(vk.DeviceSize, @sizeOf(MaterialData)) * capacity,
            .{ .storage_buffer_bit = true },
            .upload,
        );
        return .{ .buffer = buffer, .capacity = capacity, .count = 0 };
    }

    // Vulkan specification, vkDestroyBuffer: submitted work reading these
    // materials must have completed.
    pub fn deinit(self: *MaterialStorage) void {
        self.buffer.deinit();
        self.* = undefined;
    }

    // Replaces the contents. The count is asset-driven, so exceeding the
    // capacity is a runtime condition and not a programmer error: it returns an
    // error in every build rather than relying on a check the shipping build
    // removes.
    pub fn upload(self: *MaterialStorage, materials: []const MaterialData) UploadError!void {
        if (materials.len > self.capacity) return error.CapacityExceeded;
        try self.buffer.upload(std.mem.sliceAsBytes(materials));
        self.count = @intCast(materials.len);
    }

    pub fn handle(self: *const MaterialStorage) vk.Buffer {
        return self.buffer.handle;
    }

    pub fn byteSize(self: *const MaterialStorage) vk.DeviceSize {
        return @as(vk.DeviceSize, @sizeOf(MaterialData)) * self.count;
    }

    // One descriptor over the whole allocation rather than over the materials in
    // it. Uploading changes `count` and never the range, so the descriptor is
    // written once when the buffer reaches the renderer and no later upload has
    // to rewrite it. A fragment reaching past `count` reads whatever the buffer
    // was left holding, which is what the recorder's material-index validation
    // stands between.
    pub fn descriptor(self: *const MaterialStorage) vk.DescriptorBufferInfo {
        return .{
            .buffer = self.buffer.handle,
            .offset = 0,
            .range = @as(vk.DeviceSize, @sizeOf(MaterialData)) * self.capacity,
        };
    }
};

// Mirrors set 1 of scene.slang: what the scene supplies to the main pass and no
// single draw changes. The packed material array is what it holds.
//
// Its own set rather than a binding in the frame set or a second binding in each
// material set, and it sits between them by index. The three are in increasing
// order of how often they are bound: set 0 once per frame, this one once per
// recording, set 2 whenever the ordered list reaches a new material. The index
// an instance carries selects a material out of this array, so which set is
// bound is not what chooses one.
pub const bindings = [_]descriptors.Binding{
    .{ .slot = 0, .kind = .storage_buffer, .stages = .{ .fragment_bit = true } },
};

// Points the set at a material buffer. Cold: composition owns the buffer and
// hands it over once, before anything is recorded against it.
//
// The set is passed in rather than a typed wrapper around it, because the layout
// it belongs to is assembled from this list and the environment's; the type for
// that lives where the two are joined.
pub fn write(context: *const Context, set: vk.DescriptorSet, storage: *const MaterialStorage) void {
    const info = storage.descriptor();
    const writes = [_]vk.WriteDescriptorSet{.{
        .dst_set = set,
        .dst_binding = bindings[0].slot,
        .dst_array_element = 0,
        .descriptor_count = 1,
        .descriptor_type = bindings[0].kind,
        .p_image_info = &no_images,
        .p_buffer_info = @ptrCast(&info),
        .p_texel_buffer_view = &no_texel_buffers,
    }};
    context.device.updateDescriptorSets(&writes, null);
}

// Vulkan specification, VkWriteDescriptorSet: the members not selected by
// descriptorType are ignored, but the pointers are not optional in the
// structure, so they are given something valid to point at.
const no_images = [_]vk.DescriptorImageInfo{};
const no_texel_buffers = [_]vk.BufferView{};
