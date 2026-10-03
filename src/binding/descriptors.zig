const std = @import("std");
const vk = @import("vulkan");
const Context = @import("../device/context.zig").Context;
const AccelerationStructure = @import("../object/acceleration.zig").AccelerationStructure;
const Buffer = @import("../object/buffer.zig").Buffer;

const Allocator = std.mem.Allocator;

pub const InitError = error{
    ZeroSets,
} || Allocator.Error ||
    vk.DeviceWrapper.CreateDescriptorSetLayoutError ||
    vk.DeviceWrapper.CreateDescriptorPoolError ||
    vk.DeviceWrapper.AllocateDescriptorSetsError;

pub const WriteError = error{DescriptorCountMismatch};

// One entry of a set layout, as the shader declares it.
pub const Binding = struct {
    slot: u32,

    // What a write names this binding by. Required, because it is what makes a
    // source list keyed rather than positional: with seven storage buffers of
    // one type, two entries exchanged compile clean, bind a shader's velocities
    // where it reads its positions, and are reported by no layer, because both
    // buffers are real and of a plausible size. Measured on `atoms`, where the
    // swap built without a diagnostic.
    //
    // It is checked against this table and not against the shader. A name that
    // disagrees with the `.slang` file is still a name nothing here can catch;
    // what it catches is a call site that disagrees with the bindings beside it.
    name: []const u8,

    kind: vk.DescriptorType,
    stages: vk.ShaderStageFlags,

    // An array binding. One is the ordinary case and the only one a scalar
    // uniform or a single texture wants.
    count: u32 = 1,
};

// One buffer a binding is pointed at.
//
// The buffer itself rather than its handle, so that a call site naming a buffer
// names no Vulkan type at all. Nothing here validates the range against the
// buffer's size: a range past the end is what the validation layer reports, and
// a check here would only move the report earlier for callers running with the
// layer already on.
//
// The pointer is read while the write is built and never retained, so it need
// only be valid across the call. A source held past that is a source whose
// buffer may have moved, which is the hazard a handle would not have had.
pub const BufferSource = struct {
    buffer: *const Buffer,
    offset: u64 = 0,
    // The rest of the buffer from the offset when absent, which is what every
    // binding that is not sub-allocated wants.
    range: ?u64 = null,
};

// One image a binding is pointed at.
//
// The layout is the caller's to state and is not defaulted. A descriptor whose
// layout is not the one the image is actually in is a validation error at draw
// time and nothing sooner, so the value belongs where the transition that
// produced it is known.
pub const ImageSource = struct {
    view: vk.ImageView,
    sampler: vk.Sampler,
    layout: vk.ImageLayout,
};

// The handle slice is read only while the descriptor write is submitted. `one`
// points it at the owned handle inside an acceleration structure, so the common
// scalar binding exposes no Vulkan handle at its call site.
pub const AccelerationSource = struct {
    handles: []const vk.AccelerationStructureKHR,

    pub fn one(structure: *const AccelerationStructure) AccelerationSource {
        const handles: [*]const vk.AccelerationStructureKHR = @ptrCast(&structure.handle);
        return .{ .handles = handles[0..1] };
    }
};

// Vulkan specification, VkWriteDescriptorSet: the members not selected by
// descriptorType are ignored, but the pointers are not optional in the
// structure, so they are given something valid to point at.
const no_buffers = [_]vk.DescriptorBufferInfo{};
const no_images = [_]vk.DescriptorImageInfo{};
const no_texel_buffers = [_]vk.BufferView{};

// Vulkan specification, VkWriteDescriptorSet: acceleration structures are the
// fourth payload path beside its three arrays. Their handle array is carried by
// VkWriteDescriptorSetAccelerationStructureKHR through pNext. Texel buffers and
// inline uniform blocks remain unclassified until a consumer needs them.
const Family = enum { buffer, image, acceleration };

fn familyOf(kind: vk.DescriptorType) ?Family {
    return switch (kind) {
        .uniform_buffer, .storage_buffer, .uniform_buffer_dynamic, .storage_buffer_dynamic => .buffer,
        .combined_image_sampler, .sampled_image, .storage_image => .image,
        .acceleration_structure_khr => .acceleration,
        else => null,
    };
}

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
            if (binding.name.len == 0)
                @compileError("binding with no name, which nothing could then write");
            for (bindings[index + 1 ..]) |other| {
                if (binding.slot == other.slot)
                    @compileError("two bindings share a slot number");
                // Two of one name would make a keyed source list ambiguous, and
                // one of the two would silently go unwritten.
                if (std.mem.eql(u8, binding.name, other.name))
                    @compileError("two bindings share the name " ++ binding.name);
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

        // Writers select their own family from a layout. The compact source and
        // info arrays therefore contain only that family's bindings, while each
        // write retains the slot and descriptor type from the complete layout.
        // This keeps keyed writes when one set mixes buffers with an acceleration
        // structure without introducing a tagged source per binding.
        fn bindingCount(comptime wanted: Family) usize {
            var count: usize = 0;
            for (bindings) |binding| {
                if (familyOf(binding.kind) == wanted) count += 1;
            }
            return count;
        }

        const buffer_binding_count = bindingCount(.buffer);
        const image_binding_count = bindingCount(.image);
        const acceleration_binding_count = bindingCount(.acceleration);

        // What would be submitted, without submitting it. Split from the call
        // for the reason `Buffer.validateCopy` is: everything that can be wrong
        // here is decidable on the host, and a device is what otherwise stands
        // between that and a test.
        //
        // `infos` is an out parameter rather than a local because the writes
        // point into it. The two have to share a scope, and the caller is what
        // owns that scope.
        pub fn bufferWrites(
            handle: vk.DescriptorSet,
            infos: *[buffer_binding_count]vk.DescriptorBufferInfo,
            given: anytype,
        ) [buffer_binding_count]vk.WriteDescriptorSet {
            comptime if (buffer_binding_count == 0)
                @compileError("this layout has no buffer bindings");

            const sources = named(BufferSource, .buffer, given);
            var entries: [buffer_binding_count]vk.WriteDescriptorSet = undefined;
            var index: usize = 0;
            inline for (bindings) |binding| {
                if (comptime familyOf(binding.kind) != .buffer) continue;
                comptime if (binding.count != 1)
                    @compileError("a buffer array binding needs one info per element");

                const source = sources[index];
                const info = &infos[index];
                info.* = .{
                    .buffer = source.buffer.handle,
                    .offset = source.offset,
                    .range = source.range orelse vk.WHOLE_SIZE,
                };
                entries[index] = .{
                    .dst_set = handle,
                    .dst_binding = binding.slot,
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = binding.kind,
                    .p_image_info = &no_images,
                    .p_buffer_info = @ptrCast(info),
                    .p_texel_buffer_view = &no_texel_buffers,
                };
                index += 1;
            }
            return entries;
        }

        // The image half of the same shape.
        pub fn imageWrites(
            handle: vk.DescriptorSet,
            infos: *[image_binding_count]vk.DescriptorImageInfo,
            given: anytype,
        ) [image_binding_count]vk.WriteDescriptorSet {
            comptime if (image_binding_count == 0)
                @compileError("this layout has no image bindings");

            const sources = named(ImageSource, .image, given);
            var entries: [image_binding_count]vk.WriteDescriptorSet = undefined;
            var index: usize = 0;
            inline for (bindings) |binding| {
                if (comptime familyOf(binding.kind) != .image) continue;
                comptime if (binding.count != 1)
                    @compileError("an image array binding needs one info per element");

                const source = sources[index];
                const info = &infos[index];
                info.* = .{
                    .sampler = source.sampler,
                    .image_view = source.view,
                    .image_layout = source.layout,
                };
                entries[index] = .{
                    .dst_set = handle,
                    .dst_binding = binding.slot,
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = binding.kind,
                    .p_image_info = @ptrCast(info),
                    .p_buffer_info = &no_buffers,
                    .p_texel_buffer_view = &no_texel_buffers,
                };
                index += 1;
            }
            return entries;
        }

        // `infos` owns the pNext structures through the update call. A temporary
        // created inside the loop would leave each write pointing at an object
        // whose lifetime ended before vkUpdateDescriptorSets read it.
        pub fn accelerationWrites(
            handle: vk.DescriptorSet,
            infos: *[acceleration_binding_count]vk.WriteDescriptorSetAccelerationStructureKHR,
            given: anytype,
        ) WriteError![acceleration_binding_count]vk.WriteDescriptorSet {
            comptime if (acceleration_binding_count == 0)
                @compileError("this layout has no acceleration structure bindings");

            const sources = named(AccelerationSource, .acceleration, given);
            var entries: [acceleration_binding_count]vk.WriteDescriptorSet = undefined;
            var index: usize = 0;
            inline for (bindings) |binding| {
                if (comptime familyOf(binding.kind) != .acceleration) continue;

                const source = sources[index];
                if (source.handles.len != binding.count)
                    return error.DescriptorCountMismatch;
                const info = &infos[index];
                info.* = .{
                    .acceleration_structure_count = binding.count,
                    .p_acceleration_structures = source.handles.ptr,
                };
                entries[index] = .{
                    .p_next = @ptrCast(info),
                    .dst_set = handle,
                    .dst_binding = binding.slot,
                    .dst_array_element = 0,
                    .descriptor_count = binding.count,
                    .descriptor_type = binding.kind,
                    .p_image_info = &no_images,
                    .p_buffer_info = &no_buffers,
                    .p_texel_buffer_view = &no_texel_buffers,
                };
                index += 1;
            }
            return entries;
        }

        // Vulkan specification, vkUpdateDescriptorSets: the set must not be in
        // use by any submitted work that has not completed. These are therefore
        // cold paths, and pointing a set at something else while a frame is in
        // flight is the caller's problem rather than theirs.
        pub fn writeBuffers(
            self: *const Self,
            context: *const Context,
            index: usize,
            sources: anytype,
        ) void {
            var infos: [buffer_binding_count]vk.DescriptorBufferInfo = undefined;
            const entries = bufferWrites(self.set(index), &infos, sources);
            context.device.updateDescriptorSets(&entries, null);
        }

        // Where a binding of this name sits among one family's compact sources.
        fn slotOf(comptime wanted: Family, comptime name: []const u8) usize {
            var family_index: usize = 0;
            for (bindings) |binding| {
                if (familyOf(binding.kind) != wanted) continue;
                if (std.mem.eql(u8, binding.name, name)) return family_index;
                family_index += 1;
            }
            @compileError("this layout has no " ++ @tagName(wanted) ++
                " binding named " ++ name);
        }

        // A written source list, keyed by binding name, in the order the write
        // needs.
        //
        // Keyed rather than positional, which is the whole point: the slot is
        // chosen by the name, so the order a call site writes its sources in
        // means nothing and two of them exchanged is not a mistake that can be
        // made. A name the layout does not have, or one binding left unwritten,
        // is a compile error rather than a descriptor pointing at whatever the
        // neighbouring slot held.
        //
        // Bijective by construction: field names in a struct literal are
        // unique, each resolves to a distinct slot within the selected family,
        // and the count matches, so every binding of that family is written
        // exactly once and no runtime check is needed to say so.
        fn named(
            comptime Source: type,
            comptime wanted: Family,
            given: anytype,
        ) [bindingCount(wanted)]Source {
            const fields = @typeInfo(@TypeOf(given)).@"struct".fields;
            const expected = comptime bindingCount(wanted);
            comptime if (fields.len != expected) @compileError(std.fmt.comptimePrint(
                "this layout has {d} {s} bindings and the source list has {d}",
                .{ expected, @tagName(wanted), fields.len },
            ));

            var resolved: [expected]Source = undefined;
            inline for (fields) |field| {
                const source = @field(given, field.name);
                const Given = @TypeOf(source);
                const slot = &resolved[comptime slotOf(wanted, field.name)];

                if (comptime Given == Source) {
                    slot.* = source;
                } else switch (@typeInfo(Given)) {
                    // The shorthand: a buffer alone is the whole of it.
                    .pointer => {
                        comptime if (Source != BufferSource)
                            @compileError("a source is a " ++ @typeName(Source));
                        slot.* = .{ .buffer = source };
                    },
                    // Named rather than left to coercion: a field of the list
                    // has a concrete type by the time it is read here, so an
                    // anonymous literal is no longer a literal and would not
                    // coerce. A source that is not a whole buffer says its type.
                    else => @compileError(
                        "a source is a buffer or a " ++ @typeName(Source) ++
                            ", not " ++ @typeName(Given),
                    ),
                }
            }
            return resolved;
        }

        pub fn writeImages(
            self: *const Self,
            context: *const Context,
            index: usize,
            sources: anytype,
        ) void {
            var infos: [image_binding_count]vk.DescriptorImageInfo = undefined;
            const entries = imageWrites(self.set(index), &infos, sources);
            context.device.updateDescriptorSets(&entries, null);
        }

        pub fn writeAccelerationStructures(
            self: *const Self,
            context: *const Context,
            index: usize,
            sources: anytype,
        ) WriteError!void {
            var infos: [acceleration_binding_count]vk.WriteDescriptorSetAccelerationStructureKHR = undefined;
            const entries = try accelerationWrites(self.set(index), &infos, sources);
            context.device.updateDescriptorSets(&entries, null);
        }
    };
}
