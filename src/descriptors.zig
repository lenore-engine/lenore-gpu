const std = @import("std");
const vk = @import("vulkan");
const Context = @import("context.zig").Context;

const Allocator = std.mem.Allocator;

pub const InitError = error{
    ZeroSets,
} || Allocator.Error ||
    vk.DeviceWrapper.CreateDescriptorSetLayoutError ||
    vk.DeviceWrapper.CreateDescriptorPoolError ||
    vk.DeviceWrapper.AllocateDescriptorSetsError;

// One entry of a set layout, as the shader declares it.
pub const Binding = struct {
    slot: u32,
    kind: vk.DescriptorType,
    stages: vk.ShaderStageFlags,

    // An array binding. One is the ordinary case and the only one a scalar
    // uniform or a single texture wants.
    count: u32 = 1,
};

// A set layout, a pool sized for `count` copies of it, and the sets themselves.
//
// The binding list is comptime because every call site has a fixed one: it is
// the shader's interface, not a runtime choice. That is what lets the pool
// sizes be folded here rather than accumulated by hand at each site, and it
// turns a duplicated or empty binding into a compile error instead of something
// the validation layer reports on the first frame.
//
// Sets are grouped by how often they are rewritten, not by what they belong to.
// A set holding per-frame buffers is written once and bound with a dynamic
// offset per frame; a set holding a material's textures is written once per
// material and bound per draw. Putting both in one layout multiplies the two
// frequencies together, which is how one binding list becomes frames times
// batches sets.
pub fn Sets(comptime bindings: []const Binding) type {
    comptime {
        if (bindings.len == 0) @compileError("a descriptor set layout needs at least one binding");
        for (bindings, 0..) |binding, index| {
            if (binding.count == 0)
                @compileError("binding with no descriptors in it");
            for (bindings[index + 1 ..]) |other| {
                if (binding.slot == other.slot)
                    @compileError("two bindings share a slot number");
            }
        }
    }

    return struct {
        const Self = @This();

        layout: vk.DescriptorSetLayout,
        pool: vk.DescriptorPool,
        sets: []vk.DescriptorSet,

        pub const layout_bindings = blk: {
            var entries: [bindings.len]vk.DescriptorSetLayoutBinding = undefined;
            for (bindings, &entries) |binding, *entry| entry.* = .{
                .binding = binding.slot,
                .descriptor_type = binding.kind,
                .descriptor_count = binding.count,
                .stage_flags = binding.stages,
            };
            break :blk entries;
        };

        // Vulkan specification, VkDescriptorPoolSize: a pool size counts
        // descriptors, not bindings, and an array binding contributes its whole
        // length. Types are merged so that no entry can be zero, which the
        // specification forbids.
        pub const pool_sizes_per_set = blk: {
            var sizes: [bindings.len]vk.DescriptorPoolSize = undefined;
            var distinct: usize = 0;
            for (bindings) |binding| {
                var merged = false;
                for (sizes[0..distinct]) |*size| {
                    if (size.type != binding.kind) continue;
                    size.descriptor_count += binding.count;
                    merged = true;
                    break;
                }
                if (merged) continue;

                sizes[distinct] = .{ .type = binding.kind, .descriptor_count = binding.count };
                distinct += 1;
            }
            break :blk sizes[0..distinct].*;
        };

        pub fn init(context: *const Context, allocator: Allocator, count: u32) InitError!Self {
            if (count == 0) return error.ZeroSets;

            const layout = try context.device.createDescriptorSetLayout(&.{
                .binding_count = layout_bindings.len,
                .p_bindings = &layout_bindings,
            }, null);
            errdefer context.device.destroyDescriptorSetLayout(layout, null);

            var sizes = pool_sizes_per_set;
            for (&sizes) |*size| size.descriptor_count *= count;

            const pool = try context.device.createDescriptorPool(&.{
                .max_sets = count,
                .pool_size_count = sizes.len,
                .p_pool_sizes = &sizes,
            }, null);
            errdefer context.device.destroyDescriptorPool(pool, null);

            const sets = try allocator.alloc(vk.DescriptorSet, count);
            errdefer allocator.free(sets);

            // Every set has the same layout, and allocation takes one entry per
            // set rather than one shared entry.
            const layouts = try allocator.alloc(vk.DescriptorSetLayout, count);
            defer allocator.free(layouts);
            @memset(layouts, layout);

            try context.device.allocateDescriptorSets(&.{
                .descriptor_pool = pool,
                .descriptor_set_count = count,
                .p_set_layouts = layouts.ptr,
            }, sets.ptr);

            return .{ .layout = layout, .pool = pool, .sets = sets };
        }

        // Vulkan specification, vkDestroyDescriptorPool: the sets it allocated
        // are freed with it and must not be in use by a pending submission.
        pub fn deinit(self: *Self, context: *const Context, allocator: Allocator) void {
            allocator.free(self.sets);
            context.device.destroyDescriptorPool(self.pool, null);
            context.device.destroyDescriptorSetLayout(self.layout, null);
            self.* = undefined;
        }

        pub fn set(self: *const Self, index: usize) vk.DescriptorSet {
            std.debug.assert(index < self.sets.len);
            return self.sets[index];
        }
    };
}
