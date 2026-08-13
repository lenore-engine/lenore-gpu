const std = @import("std");

const mesh_module = @import("../object/mesh.zig");
const pool_module = @import("pool.zig");
const owning = @import("owning.zig");
const texture_cache = @import("../object/texture_cache.zig");

const Allocator = std.mem.Allocator;
const Bound = texture_cache.Bound;
const Mesh = mesh_module.Mesh;

// The five texture slots a material binds, in the order the packed material
// layout indexes them. A slot without a texture of its own holds the neutral
// fallback rather than nothing, so a shader samples all five unconditionally.
//
// Bound values, not references into the texture cache: what a descriptor needs
// is a view and a sampler, and both outlive any rehash of the cache.
pub const TextureSet = struct {
    base_colour: Bound,
    metallic_roughness: Bound,
    normal: Bound,
    emissive: Bound,
    occlusion: Bound,
};

pub const MeshHandle = owning.OwningStorage(Mesh).Handle;
pub const TextureSetHandle = pool_module.ResourcePool(TextureSet).Handle;

// The scene's resource tables.
//
// Meshes are owned here and destroyed with the storage, because nothing else
// holds them. Texture sets are not: every image inside one belongs to the
// texture cache by content key and is released there, so a set is only a
// resolved binding and dropping it frees nothing.
//
// That asymmetry is why the two use different containers rather than one
// generic table.
pub const ResourceStorage = struct {
    meshes: owning.OwningStorage(Mesh),
    texture_sets: pool_module.ResourcePool(TextureSet),

    pub const empty: ResourceStorage = .{ .meshes = .empty, .texture_sets = .empty };

    // Vulkan specification, vkDestroyBuffer: submitted work drawing these meshes
    // must have completed.
    pub fn deinit(self: *ResourceStorage, allocator: Allocator) void {
        self.meshes.deinit(allocator);
        self.texture_sets.deinit(allocator);
        self.* = undefined;
    }

    // Takes ownership of the mesh, destroying it if the slot cannot be reserved.
    pub fn addMesh(
        self: *ResourceStorage,
        allocator: Allocator,
        owned: Mesh,
    ) pool_module.AddError!MeshHandle {
        return self.meshes.add(allocator, owned);
    }

    pub fn addTextureSet(
        self: *ResourceStorage,
        allocator: Allocator,
        set: TextureSet,
    ) pool_module.AddError!TextureSetHandle {
        return self.texture_sets.add(allocator, set);
    }

    pub fn mesh(self: *const ResourceStorage, handle: MeshHandle) ?*const Mesh {
        return self.meshes.get(handle);
    }

    pub fn textureSet(self: *const ResourceStorage, handle: TextureSetHandle) ?*const TextureSet {
        return self.texture_sets.get(handle);
    }

    // Destroys the mesh behind the handle. A stale handle is a no-op.
    pub fn removeMesh(self: *ResourceStorage, handle: MeshHandle) void {
        self.meshes.remove(handle);
    }

    // Frees the slot. Nothing GPU-side is destroyed: the images belong to the
    // texture cache, which the caller releases by key.
    pub fn removeTextureSet(self: *ResourceStorage, handle: TextureSetHandle) void {
        _ = self.texture_sets.remove(handle);
    }

    pub fn meshCount(self: *const ResourceStorage) u32 {
        return self.meshes.count();
    }

    pub fn textureSetCount(self: *const ResourceStorage) u32 {
        return self.texture_sets.count();
    }
};
