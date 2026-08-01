const std = @import("std");
const SamplerConfig = @import("sampler.zig").SamplerConfig;

const Allocator = std.mem.Allocator;

// The description of a material as it comes out of an asset, before anything
// GPU-side exists for it. It holds no Vulkan object and no handle: what it names
// is content, and resolving those names to images and samplers is the upload
// path's work.
//
// Defaults follow the glTF 2.0 specification, section Materials, so a material
// that omits a property behaves as the format says it should rather than as
// whatever this struct happened to zero-initialize.
pub const MaterialInfo = struct {
    // Owned. Diagnostics only: identity for lookup is a texture's content key.
    name: []const u8,
    textures: TextureMaps,
    factors: Factors,
    rendering: Rendering,

    pub const TextureMaps = struct {
        base_colour: Slot = .{},
        metallic_roughness: Slot = .{},
        normal: Slot = .{},
        emissive: Slot = .{},
        occlusion: Slot = .{},

        pub const Slot = struct {
            // Owned when present. Null means the slot has no texture and the
            // material falls back to its factors.
            path: ?[]const u8 = null,
            sampler: SamplerConfig = .{},
            uv: UvTransform = .{},
        };

        // glTF 2.0, KHR_texture_transform, plus the UV set selector. Stored as
        // the extension states it; folding offset, rotation and scale into the
        // matrix a shader wants belongs to whoever packs the GPU layout, not
        // here.
        pub const UvTransform = struct {
            set: u32 = 0,
            offset: [2]f32 = .{ 0.0, 0.0 },
            rotation: f32 = 0.0,
            scale: [2]f32 = .{ 1.0, 1.0 },
        };

        pub fn deinit(self: *TextureMaps, allocator: Allocator) void {
            inline for (@typeInfo(TextureMaps).@"struct".fields) |field| {
                if (@field(self, field.name).path) |path| allocator.free(path);
            }
            self.* = undefined;
        }
    };

    // Multiplied with the corresponding texture where one is present, and used
    // alone where it is not.
    pub const Factors = struct {
        base_colour: [4]f32 = .{ 1.0, 1.0, 1.0, 1.0 },
        metallic: f32 = 1.0,
        roughness: f32 = 1.0,
        emissive: [3]f32 = .{ 0.0, 0.0, 0.0 },
        // Scales the tangent-space X and Y of a sampled normal. It does nothing
        // without a normal map.
        normal_scale: f32 = 1.0,
        // Interpolates between no occlusion and the sampled value.
        occlusion_strength: f32 = 1.0,
    };

    pub const Rendering = struct {
        // The glTF names, with the Zig keyword escaped rather than prefixed.
        pub const AlphaMode = enum { @"opaque", mask, blend };

        alpha_mode: AlphaMode = .@"opaque",
        // Read only in mask mode, where a sample below it is discarded.
        alpha_cutoff: f32 = 0.5,
        double_sided: bool = false,
        // glTF 2.0, KHR_materials_unlit: the material is displayed as its base
        // colour with no lighting applied.
        unlit: bool = false,
    };

    pub fn deinit(self: *MaterialInfo, allocator: Allocator) void {
        allocator.free(self.name);
        self.textures.deinit(allocator);
        self.* = undefined;
    }
};
