const std = @import("std");
const vk = @import("vulkan");

const Context = @import("../device/context.zig").Context;
const memory = @import("../memory/allocator.zig");
const mesh_module = @import("../object/mesh.zig");
const pool_module = @import("../store/pool.zig");
const resource_storage = @import("../store/resources.zig");
const sampler_module = @import("../object/sampler.zig");
const staging = @import("pool.zig");
const texture_cache = @import("../object/texture_cache.zig");
const transfer_module = @import("transfer.zig");

const Allocator = std.mem.Allocator;
const Bound = texture_cache.Bound;
const Mesh = mesh_module.Mesh;
const MeshHandle = resource_storage.MeshHandle;
const ResourceStorage = resource_storage.ResourceStorage;
const SamplerConfig = @import("lenore-resources").SamplerConfig;
const StagingPool = staging.StagingPool;
const TextureCache = texture_cache.TextureCache;
const Transfer = transfer_module.Transfer;
const TextureSet = resource_storage.TextureSet;
const TextureSetHandle = resource_storage.TextureSetHandle;

// The two GPU-ready texture sources. A KTX2 carries its own format and mip
// chain; decoded RGBA8 carries one tightly packed level and receives the format
// required by the material slot.
pub const TextureSource = union(enum) {
    ktx2: []const u8,
    rgba8: texture_cache.Rgba8,
};

// One texture a material asks for: the source image identity, its GPU-ready
// bytes, and how it wants them sampled. The upload path includes the selected
// Vulkan format in the cache key, because the same decoded image interpreted as
// sRGB colour and linear data needs two immutable image formats.
pub const TextureRequest = struct {
    key: []const u8,
    source: TextureSource,
    sampler: SamplerConfig = .{},
};

// What fills one slot of a set: bytes this batch is to upload, or an image it
// has already uploaded through addTexture.
//
// The second exists because a source image and a material slot are not the same
// unit. Several materials name one image, and a loader that walks materials
// decodes and uploads it once per naming. Walking images first and binding
// second is what makes each image cost one decode, and this is what the second
// pass has to say.
pub const TextureSlot = union(enum) {
    request: TextureRequest,
    // The sampler is the use's own. It is not the one addTexture was given,
    // because in glTF a sampler belongs to the texture that references an image
    // rather than to the image, so two references may filter it differently.
    resident: struct {
        texture: texture_cache.Resident,
        sampler: SamplerConfig = .{},
    },
};

// The five slots, each either a texture to load or nothing, in which case the
// slot binds its neutral fallback.
pub const TextureSetRequest = struct {
    base_colour: ?TextureSlot = null,
    metallic_roughness: ?TextureSlot = null,
    normal: ?TextureSlot = null,
    emissive: ?TextureSlot = null,
    occlusion: ?TextureSlot = null,
};

// What a slot's data must be, and what stands in when it is absent. Colour is
// sRGB and everything else is linear, because a roughness read through an sRGB
// transfer function is a wrong number rather than a wrong shade.
//
// Public because addTexture uploads one texture outside a set and the format is
// this policy, not the caller's: a loader says which slot the image is for and
// the interpretation follows.
pub const MaterialSlot = enum {
    base_colour,
    metallic_roughness,
    normal,
    emissive,
    occlusion,

    fn format(self: MaterialSlot, source: TextureSource) vk.Format {
        return switch (source) {
            .ktx2 => switch (self) {
                .base_colour, .emissive => .bc7_srgb_block,
                .metallic_roughness, .normal, .occlusion => .bc7_unorm_block,
            },
            .rgba8 => switch (self) {
                .base_colour, .emissive => .r8g8b8a8_srgb,
                .metallic_roughness, .normal, .occlusion => .r8g8b8a8_unorm,
            },
        };
    }

    fn fallback(self: MaterialSlot) texture_cache.Fallback {
        return switch (self) {
            .base_colour, .emissive => .white,
            .metallic_roughness => .metallic_roughness,
            .normal => .normal,
            .occlusion => .white,
        };
    }
};

comptime {
    // The request, the bound set and the slot list describe the same five slots
    // under the same names. A slot added to one and not the others would
    // otherwise be silently dropped at upload.
    const slots = @typeInfo(MaterialSlot).@"enum".fields;
    std.debug.assert(slots.len == @typeInfo(TextureSetRequest).@"struct".fields.len);
    std.debug.assert(slots.len == @typeInfo(TextureSet).@"struct".fields.len);
    for (slots) |slot| {
        std.debug.assert(@hasField(TextureSetRequest, slot.name));
        std.debug.assert(@hasField(TextureSet, slot.name));
    }
}

// Cache identity is source identity plus immutable image format. A fixed-width
// suffix makes the pairing injective without escaping arbitrary path bytes.
fn interpretedKey(
    allocator: Allocator,
    source_key: []const u8,
    format: vk.Format,
) (Allocator.Error || error{TextureKeyTooLong})![]u8 {
    const length = std.math.add(usize, source_key.len, @sizeOf(u32)) catch
        return error.TextureKeyTooLong;
    const key = try allocator.alloc(u8, length);
    @memcpy(key[0..source_key.len], source_key);
    std.mem.writeInt(
        u32,
        key[source_key.len..][0..@sizeOf(u32)],
        @intCast(@intFromEnum(format)),
        .little,
    );
    return key;
}

pub const BeginError = transfer_module.BeginError || Allocator.Error;
pub const AddMeshError = mesh_module.InitError || pool_module.AddError;
pub const AddTextureError = error{TextureKeyTooLong} ||
    texture_cache.AcquireError || sampler_module.GetError;
pub const AddTextureSetError = AddTextureError || pool_module.AddError;
pub const FinishError = transfer_module.FinishError;

// What one completed batch put into the storage and the cache. It exists so a
// scene can hand back exactly what it took, and no more: releasing a texture key
// that some other scene also holds only drops that scene's reference.
pub const Uploaded = struct {
    // The lists themselves rather than their contents. std.ArrayList keeps items
    // as a prefix of a larger allocation whenever capacity exceeds length, so
    // freeing that slice would hand the allocator a size it never gave out.
    meshes: std.ArrayList(MeshHandle),
    texture_sets: std.ArrayList(TextureSetHandle),
    texture_keys: std.ArrayList([]const u8),

    // Returns every resource this batch registered. Emptying the lists is what
    // makes a second call do nothing, so a scene that unloads twice does not
    // release a reference it no longer holds.
    pub fn release(
        self: *Uploaded,
        allocator: Allocator,
        storage: *ResourceStorage,
        cache: *TextureCache,
    ) void {
        for (self.meshes.items) |handle| storage.removeMesh(handle);
        for (self.texture_sets.items) |handle| storage.removeTextureSet(handle);
        for (self.texture_keys.items) |key| {
            cache.release(key);
            allocator.free(key);
        }
        self.meshes.clearRetainingCapacity();
        self.texture_sets.clearRetainingCapacity();
        self.texture_keys.clearRetainingCapacity();
    }

    // Frees the bookkeeping, and whatever the scene still held if release was
    // never called.
    pub fn deinit(
        self: *Uploaded,
        allocator: Allocator,
        storage: *ResourceStorage,
        cache: *TextureCache,
    ) void {
        self.release(allocator, storage, cache);
        self.meshes.deinit(allocator);
        self.texture_sets.deinit(allocator);
        self.texture_keys.deinit(allocator);
        self.* = undefined;
    }
};

// A failure-atomic upload transaction.
//
// Dropping a batch that was never finished retires its transfer first and only
// then destroys what it registered, because commands must not name a destroyed
// resource: a command buffer holding such a command cannot even be ended. That
// ordering is the reason this type exists rather than a loop over individual
// uploads.
//
// The batch holds every texture reference it takes until finish returns, for the
// same reason. Releasing one earlier destroys an image a recorded copy still
// names.
//
// What it no longer promises is a single submission. The transfer submits and
// waits whenever the staging pool has to be reclaimed, so a large batch reaches
// the device in several pieces. Atomicity is unaffected: every piece has
// completed before the next is recorded, so rolling back is still only a
// question of what may still be pending, and that is what the transfer answers.
pub const Batch = struct {
    allocator: Allocator,
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    storage: *ResourceStorage,
    cache: *TextureCache,
    transfer: Transfer,
    // Set by any failed registration. A batch that has partly failed cannot be
    // finished, because what it recorded no longer matches what it registered.
    failed: bool,

    meshes: std.ArrayList(MeshHandle),
    texture_sets: std.ArrayList(TextureSetHandle),
    texture_keys: std.ArrayList([]const u8),

    pub fn begin(
        allocator: Allocator,
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        storage: *ResourceStorage,
        cache: *TextureCache,
        staging_pool: *StagingPool,
        pool: vk.CommandPool,
    ) BeginError!Batch {
        return .{
            .allocator = allocator,
            .context = context,
            .memory_allocator = memory_allocator,
            .storage = storage,
            .cache = cache,
            .transfer = try .begin(context, pool, staging_pool),
            .failed = false,
            .meshes = .empty,
            .texture_sets = .empty,
            .texture_keys = .empty,
        };
    }

    // Rolls the batch back. Retiring the transfer comes first: its command
    // buffer holds commands naming the resources released below, and Vulkan
    // requires those to outlive it.
    //
    // A batch whose submission failed after reaching the queue is the one case
    // that cannot be rolled back at all. Its buffer may be pending, so neither
    // the buffer nor anything its commands name may be destroyed, and nothing
    // available here proves otherwise. Everything is abandoned and reported: a
    // leak is recoverable and a use-after-free is not.
    pub fn deinit(self: *Batch) void {
        const abandoned = self.transfer.abandoned();
        self.transfer.deinit();
        if (abandoned) {
            std.log.err(
                "upload batch abandoned after a failed submission: {d} meshes " ++
                    "and {d} texture sets are leaked because the submission may " ++
                    "still be pending",
                .{ self.meshes.items.len, self.texture_sets.items.len },
            );
        } else {
            self.releaseRegistered();
        }

        self.meshes.deinit(self.allocator);
        self.texture_sets.deinit(self.allocator);
        for (self.texture_keys.items) |key| self.allocator.free(key);
        self.texture_keys.deinit(self.allocator);
        self.* = undefined;
    }

    fn releaseRegistered(self: *Batch) void {
        for (self.meshes.items) |handle| self.storage.removeMesh(handle);
        for (self.texture_sets.items) |handle| self.storage.removeTextureSet(handle);
        for (self.texture_keys.items) |key| self.cache.release(key);
    }

    // Packs and records a mesh, and registers it with the storage. The mesh is
    // destroyed rather than leaked if registration fails, and the batch is
    // poisoned so that finish cannot submit a command buffer describing work for
    // a resource nobody holds.
    pub fn addMesh(
        self: *Batch,
        comptime IndexType: type,
        upload: mesh_module.Upload(IndexType),
    ) AddMeshError!MeshHandle {
        std.debug.assert(self.transfer.state == .recording);
        errdefer self.failed = true;

        try self.meshes.ensureUnusedCapacity(self.allocator, 1);
        const mesh = try Mesh.init(
            IndexType,
            self.context,
            self.memory_allocator,
            &self.transfer,
            upload,
        );
        const handle = try self.storage.addMesh(self.allocator, mesh);
        self.meshes.appendAssumeCapacity(handle);
        return handle;
    }

    // Resolves the five slots, loading what is absent from the cache and binding
    // the neutral fallback where a request supplies nothing.
    // Uploads one texture on its own and hands back the image a later set binds.
    // The batch holds that reference until the caller releases what it took,
    // exactly as for a texture named inside a set.
    //
    // This is what a loader that walks source images before materials calls. The
    // sets it builds afterwards fill their slots with `.resident`, so an image
    // several materials name is decoded once and uploaded once per format it is
    // interpreted in.
    pub fn addTexture(
        self: *Batch,
        comptime slot: MaterialSlot,
        request: TextureRequest,
    ) AddTextureError!Bound {
        std.debug.assert(self.transfer.state == .recording);
        errdefer self.failed = true;

        try self.texture_keys.ensureUnusedCapacity(self.allocator, 1);
        return self.resolveSlot(slot, .{ .request = request });
    }

    pub fn addTextureSet(
        self: *Batch,
        request: TextureSetRequest,
    ) AddTextureSetError!TextureSetHandle {
        std.debug.assert(self.transfer.state == .recording);
        errdefer self.failed = true;

        try self.texture_sets.ensureUnusedCapacity(self.allocator, 1);
        try self.texture_keys.ensureUnusedCapacity(
            self.allocator,
            @typeInfo(MaterialSlot).@"enum".fields.len,
        );

        var set: TextureSet = undefined;
        inline for (@typeInfo(MaterialSlot).@"enum".fields) |field| {
            const slot = @field(MaterialSlot, field.name);
            @field(set, field.name) = try self.resolveSlot(
                slot,
                @field(request, field.name),
            );
        }

        const handle = try self.storage.addTextureSet(self.allocator, set);
        self.texture_sets.appendAssumeCapacity(handle);
        return handle;
    }

    // Submits the last of the recorded work and waits for it. On success the
    // batch owns nothing: the storage holds the resources and the caller holds
    // the record of what to release.
    //
    // A failure here leaves the transfer to say whether its buffer may be
    // pending, and the caller's deinit is what asks.
    pub fn finish(self: *Batch) FinishError!Uploaded {
        std.debug.assert(self.transfer.state == .recording);
        std.debug.assert(!self.failed);

        try self.transfer.finish();
        self.transfer.deinit();

        const uploaded: Uploaded = .{
            .meshes = self.meshes,
            .texture_sets = self.texture_sets,
            .texture_keys = self.texture_keys,
        };
        self.meshes = .empty;
        self.texture_sets = .empty;
        self.texture_keys = .empty;
        self.* = undefined;
        return uploaded;
    }

    fn resolveSlot(
        self: *Batch,
        comptime slot: MaterialSlot,
        filling: ?TextureSlot,
    ) AddTextureError!Bound {
        const filled = filling orelse
            return self.cache.fallback(slot.fallback(), .{});

        const wanted = switch (filled) {
            // No key is recorded: the reference that keeps this image alive was
            // taken when addTexture uploaded it, and releasing a set must not
            // drop it twice.
            .resident => |held| return held.texture.bind(
                try self.cache.sampler(held.sampler),
            ),
            .request => |request| request,
        };

        const format = slot.format(wanted.source);
        // The batch owns the interpreted key because it outlives the request,
        // and releasing on rollback needs the exact identity the cache received.
        const key = try interpretedKey(self.allocator, wanted.key, format);
        errdefer self.allocator.free(key);

        const bound = switch (wanted.source) {
            .ktx2 => |bytes| try self.cache.acquireKtx2(
                key,
                bytes,
                format,
                // Every slot of a material texture set is a 2D sampler. An
                // environment is not acquired through here.
                .texture_2d,
                wanted.sampler,
                &self.transfer,
            ),
            .rgba8 => |source| try self.cache.acquireRgba8(
                key,
                source,
                format,
                wanted.sampler,
                &self.transfer,
            ),
        };
        // Recorded before the reference is registered, so a failure between the
        // two would drop a reference nobody releases. Appending cannot fail: the
        // capacity for every slot was reserved before the first was resolved.
        self.texture_keys.appendAssumeCapacity(key);
        return bound;
    }
};
