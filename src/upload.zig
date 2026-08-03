const std = @import("std");
const vk = @import("vulkan");

const commands = @import("commands.zig");
const Context = @import("context.zig").Context;
const memory = @import("memory/allocator.zig");
const mesh_module = @import("mesh/resource.zig");
const pool_module = @import("pool.zig");
const resource_storage = @import("resource_storage.zig");
const sampler_module = @import("sampler.zig");
const staging = @import("staging/arena.zig");
const texture_cache = @import("texture_cache.zig");

const Allocator = std.mem.Allocator;
const Bound = texture_cache.Bound;
const Mesh = mesh_module.Mesh;
const MeshHandle = resource_storage.MeshHandle;
const ResourceStorage = resource_storage.ResourceStorage;
const SamplerConfig = @import("lenore-resources").SamplerConfig;
const StagingArena = staging.StagingArena;
const TextureCache = texture_cache.TextureCache;
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

// The five slots, each either a texture to load or nothing, in which case the
// slot binds its neutral fallback.
pub const TextureSetRequest = struct {
    base_colour: ?TextureRequest = null,
    metallic_roughness: ?TextureRequest = null,
    normal: ?TextureRequest = null,
    emissive: ?TextureRequest = null,
    occlusion: ?TextureRequest = null,
};

// What a slot's data must be, and what stands in when it is absent. Colour is
// sRGB and everything else is linear, because a roughness read through an sRGB
// transfer function is a wrong number rather than a wrong shade.
const Slot = enum {
    base_colour,
    metallic_roughness,
    normal,
    emissive,
    occlusion,

    fn format(self: Slot, source: TextureSource) vk.Format {
        return switch (source) {
            .ktx2 => switch (self) {
                .base_colour, .emissive => .bc7_srgb_block,
                .metallic_roughness, .occlusion => .bc7_unorm_block,
                // Two channels are enough for a tangent-space normal: the third
                // is reconstructed from them.
                .normal => .bc5_unorm_block,
            },
            .rgba8 => switch (self) {
                .base_colour, .emissive => .r8g8b8a8_srgb,
                .metallic_roughness, .normal, .occlusion => .r8g8b8a8_unorm,
            },
        };
    }

    fn fallback(self: Slot) texture_cache.Fallback {
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
    const slots = @typeInfo(Slot).@"enum".fields;
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

pub const BeginError = commands.BeginError || Allocator.Error;
pub const AddMeshError = mesh_module.InitError || pool_module.AddError;
pub const AddTextureSetError = error{TextureKeyTooLong} ||
    texture_cache.AcquireError || sampler_module.GetError || pool_module.AddError;
pub const FinishError = commands.SubmitError;

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
// Every copy is recorded into one command buffer and submitted once by finish.
// Dropping a batch that was never finished frees that command buffer first and
// only then destroys what it registered, because commands must not name a
// destroyed resource: a buffer holding such a command cannot even be ended.
// That ordering is the reason this type exists rather than a loop over
// individual uploads.
//
// The batch holds every texture reference it takes until finish returns, for the
// same reason. Releasing one earlier destroys an image the recorded copy still
// names.
// Three states, because ownership of the command buffer and permission to free
// it are not the same thing.
//
// Vulkan specification, vkFreeCommandBuffers: a command buffer must not be freed
// while it may still be pending. A submission that fails after the work reached
// the queue establishes neither completion nor that it never started, so the
// buffer stays ours and stays unfreeable. Collapsing that into one boolean is
// what made an earlier version free a possibly pending buffer.
const CommandState = enum {
    // Ours, not submitted, safe to free.
    recording,
    // Submitted with an unknown outcome. Freeing it, and destroying anything its
    // commands name, is forbidden until completion or device loss is
    // established, and nothing here can establish either.
    pending,
    // Submitted and waited for. Already freed by the submission path.
    consumed,
};

pub const Batch = struct {
    allocator: Allocator,
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    storage: *ResourceStorage,
    cache: *TextureCache,
    arena: *StagingArena,
    pool: vk.CommandPool,
    command_buffer: vk.CommandBuffer,
    command_state: CommandState,
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
        arena: *StagingArena,
        pool: vk.CommandPool,
    ) BeginError!Batch {
        const command_buffer = try commands.beginOneShot(context, pool);
        return .{
            .allocator = allocator,
            .context = context,
            .memory_allocator = memory_allocator,
            .storage = storage,
            .cache = cache,
            .arena = arena,
            .pool = pool,
            .command_buffer = command_buffer,
            .command_state = .recording,
            .failed = false,
            .meshes = .empty,
            .texture_sets = .empty,
            .texture_keys = .empty,
        };
    }

    // Rolls the batch back. Freeing the command buffer comes first: it holds
    // commands naming the resources destroyed below, and Vulkan requires those
    // to outlive it.
    //
    // A batch whose submission failed after reaching the queue is the one case
    // that cannot be rolled back at all. Its buffer may be pending, so neither
    // the buffer nor anything its commands name may be destroyed, and nothing
    // available here proves otherwise. Everything is abandoned and reported: a
    // leak is recoverable and a use-after-free is not.
    pub fn deinit(self: *Batch) void {
        switch (self.command_state) {
            .recording => {
                self.context.device.freeCommandBuffers(self.pool, &.{self.command_buffer});
                self.releaseRegistered();
            },
            .consumed => self.releaseRegistered(),
            .pending => std.log.err(
                "upload batch abandoned after a failed submission: {d} meshes, " ++
                    "{d} texture sets and one command buffer are leaked because " ++
                    "the submission may still be pending",
                .{ self.meshes.items.len, self.texture_sets.items.len },
            ),
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
        std.debug.assert(self.command_state == .recording);
        errdefer self.failed = true;

        try self.meshes.ensureUnusedCapacity(self.allocator, 1);
        const mesh = try Mesh.init(
            IndexType,
            self.context,
            self.memory_allocator,
            self.arena,
            self.command_buffer,
            upload,
        );
        const handle = try self.storage.addMesh(self.allocator, mesh);
        self.meshes.appendAssumeCapacity(handle);
        return handle;
    }

    // Resolves the five slots, loading what is absent from the cache and binding
    // the neutral fallback where a request supplies nothing.
    pub fn addTextureSet(
        self: *Batch,
        request: TextureSetRequest,
    ) AddTextureSetError!TextureSetHandle {
        std.debug.assert(self.command_state == .recording);
        errdefer self.failed = true;

        try self.texture_sets.ensureUnusedCapacity(self.allocator, 1);
        try self.texture_keys.ensureUnusedCapacity(
            self.allocator,
            @typeInfo(Slot).@"enum".fields.len,
        );

        var set: TextureSet = undefined;
        inline for (@typeInfo(Slot).@"enum".fields) |field| {
            const slot = @field(Slot, field.name);
            @field(set, field.name) = try self.resolveSlot(
                slot,
                @field(request, field.name),
            );
        }

        const handle = try self.storage.addTextureSet(self.allocator, set);
        self.texture_sets.appendAssumeCapacity(handle);
        return handle;
    }

    // Ends the command buffer, submits it and waits. On success the batch owns
    // nothing: the storage holds the resources and the caller holds the record
    // of what to release.
    //
    // The state moves before the call, not after. A failure inside it may leave
    // the buffer pending, and a rollback that assumed otherwise would free it
    // and destroy what its commands name.
    pub fn finish(self: *Batch) FinishError!Uploaded {
        std.debug.assert(self.command_state == .recording);
        std.debug.assert(!self.failed);

        self.command_state = .pending;
        try commands.submitOneShotAndWait(self.context, self.pool, self.command_buffer);
        self.command_state = .consumed;

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
        comptime slot: Slot,
        request: ?TextureRequest,
    ) AddTextureSetError!Bound {
        const wanted = request orelse
            return self.cache.fallback(slot.fallback(), .{});

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
                wanted.sampler,
                self.arena,
                self.command_buffer,
            ),
            .rgba8 => |source| try self.cache.acquireRgba8(
                key,
                source,
                format,
                wanted.sampler,
                self.arena,
                self.command_buffer,
            ),
        };
        // Recorded before the reference is registered, so a failure between the
        // two would drop a reference nobody releases. Appending cannot fail: the
        // capacity for every slot was reserved before the first was resolved.
        self.texture_keys.appendAssumeCapacity(key);
        return bound;
    }
};
