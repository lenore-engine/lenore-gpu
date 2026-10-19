const std = @import("std");
const vk = @import("vulkan");
const zm = @import("zmath");
const Context = @import("../device/context.zig").Context;
const descriptors = @import("descriptors.zig");
const memory = @import("../memory/allocator.zig");
const per_frame = @import("per_frame.zig");
const uniforms = @import("uniforms.zig");

const Allocator = std.mem.Allocator;

// What the main pass reads that changes once per frame, and nothing else.
//
// Every ring is a dynamic descriptor, so the set is created once and the frame
// is chosen by the offsets passed at bind time. That is what keeps the set
// count off the frame axis: a set per frame would have to be written again for
// every ring added.
//
// Set 0, and the whole of what a shader may read out of it.
//
// Each entry names the ring behind it as well as the binding, in one literal.
// Both the descriptor write and the dynamic offsets are derived from this, so
// they cannot disagree about which ring a slot holds. Vulkan takes dynamic
// offsets in increasing binding order, and getting that order wrong hands one
// ring the other's offset without failing anything.
// The order here is the order `init` builds its descriptor infos in, which is
// what lets an entry select its own by the tag alone.
const Ring = enum { camera, instances, lights, joints, sun_shadow };

const Entry = struct {
    binding: descriptors.Binding,
    ring: Ring,
};

const entries = [_]Entry{
    .{
        // Both stages: the vertex stage transforms by the view-projection and
        // the fragment stage reads the eye for the view-dependent half of the
        // BRDF.
        .binding = .{
            .slot = 0,
            .name = "camera",
            .kind = .uniform_buffer_dynamic,
            .stages = .{ .vertex_bit = true, .fragment_bit = true },
        },
        .ring = .camera,
    },
    .{
        .binding = .{ .slot = 1, .name = "instances", .kind = .storage_buffer_dynamic, .stages = .{ .vertex_bit = true } },
        .ring = .instances,
    },
    .{
        .binding = .{ .slot = 2, .name = "lights", .kind = .uniform_buffer_dynamic, .stages = .{ .fragment_bit = true } },
        .ring = .lights,
    },
    .{
        // A storage buffer rather than a uniform one: 16384 bytes is all
        // `maxUniformBufferRange` guarantees (vk.xml), which is 256 matrices for
        // a whole scene, and the joint array is per scene and not per skeleton.
        .binding = .{ .slot = 3, .name = "joints", .kind = .storage_buffer_dynamic, .stages = .{ .vertex_bit = true } },
        .ring = .joints,
    },
    .{
        // The main pass's fragment stage transforms the shaded point by this
        // matrix to find the texel to compare against; the bake rasterizes
        // through the same matrix, pushed to it. The vertex stage stays in the
        // flags because the bake's pipeline layout names this set, and a stage
        // that reads nothing from a binding costs nothing.
        .binding = .{
            .slot = 4,
            .name = "sun_shadow",
            .kind = .uniform_buffer_dynamic,
            .stages = .{ .vertex_bit = true, .fragment_bit = true },
        },
        .ring = .sun_shadow,
    },
};

pub const bindings = blk: {
    var out: [entries.len]descriptors.Binding = undefined;
    for (entries, &out) |entry, *binding| binding.* = entry.binding;
    break :blk out;
};

comptime {
    // The order the offsets are handed over in is the table's order, so the
    // table has to be the order Vulkan reads them in.
    for (entries[1..], entries[0 .. entries.len - 1]) |next, previous| {
        if (next.binding.slot <= previous.binding.slot)
            @compileError("frame set bindings are not in increasing slot order");
    }
}

// The buffer usage a binding of that kind needs. Derived rather than written
// beside each ring: a ring created without the usage its descriptor declares is
// a mismatch the validation layer reports and nothing else does.
fn usageFor(comptime kind: vk.DescriptorType) vk.BufferUsageFlags {
    return switch (kind) {
        .uniform_buffer, .uniform_buffer_dynamic => .{ .uniform_buffer_bit = true },
        .storage_buffer, .storage_buffer_dynamic => .{ .storage_buffer_bit = true },
        else => @compileError("frame set binding is not a buffer"),
    };
}

fn entryFor(comptime ring: Ring) Entry {
    for (entries) |entry| {
        if (entry.ring == ring) return entry;
    }
    @compileError("frame set ring with no binding in front of it");
}

const Sets = descriptors.Sets(&bindings);

pub const InitError = per_frame.InitError || per_frame.LayoutError || descriptors.InitError;

pub const UpdateError = error{
    // More instances than the ring was built for. A runtime condition, not a
    // programmer error: how much a scene holds is the scene's business.
    InstanceCapacityExceeded,

    // More lights than the block holds. Same reasoning, but the bound is
    // `uniforms.max_lights` rather than a size chosen at init: the shader's
    // array is fixed, so widening it is a change to both sides.
    LightCapacityExceeded,

    // More joint matrices than the ring was built for. The scene's own plan
    // reports this first, against the same bound; this is the check that stands
    // between the arithmetic and the write.
    JointCapacityExceeded,
    PreviousJointCountMismatch,

    // More local shadow matrices than the block holds (`uniforms.max_shadow_layers`).
    ShadowLayerCapacityExceeded,

    // A light names a shadow layer it cannot have: a kind with no local map,
    // or a run of layers (one for a spot, six for a point light) that ends
    // past the frame's matrices. The shader indexes the matrices by this
    // number without a check, so it is refused here instead.
    ShadowLayerOutOfRange,
};

// What one instance carries, as the shader's `StructuredBuffer<Instance>` steps
// through it. A separate name from `zm.Mat` because this is a GPU layout:
// widening it is a change to the shader, and widening `zm.Mat` is not.
pub const Instance = extern struct {
    model: uniforms.Mat4 align(16),

    // Where this instance's joint matrices begin in the frame's joint array.
    // Read only by the skinned pipeline variant, which is the only one that
    // indexes that array.
    joint_base: u32,

    // The material and two padding words are separate scalars because the
    // shader's mirror is held against this one by field name. A three-component
    // vector would move those fields and the previous matrix away from the
    // offsets the host writes. Reflection checks both the fields and stride.
    material_index: u32 = 0,
    padding_0: u32 = 0,
    padding_1: u32 = 0,

    // Last frame's `model`, for the motion the depth prepass writes. After the
    // fields above rather than beside `model`, so that a shader reading the
    // record's first eighty bytes reads them where it always did.
    previous_model: uniforms.Mat4 align(16),
};

// One joint matrix, as `StructuredBuffer<float4x4>` reads it.
//
// Sixty-four bytes rather than a packed forty-eight. glTF 2.0 section 3.7.3.1
// requires the fourth row of every inverse bind matrix to be [0,0,0,1], and a
// pose composes TRS, so a joint matrix is affine and the packed form would be
// correct. It is not taken because in the row-vector convention the unused lane
// is the fourth of each row rather than a contiguous tail: packing turns a copy
// of whole rows into a gather of three lanes from each, and at the joint counts
// a single character carries that trades a measurable cost for a saving nothing
// has missed.
pub const Joint = uniforms.Mat4;

// How much of each ring one frame's slot holds.
pub const Capacity = struct {
    instances: usize,
    joints: usize,
};

// Whether a frame's contents fit the slot they are about to be written into.
//
// Split from `update` for the reason `buffer.validateCopy` is: everything that
// can be wrong in it is arithmetic, and a device is what stands between that
// arithmetic and a test. Every bound is answered before anything is written, so
// a frame that does not fit leaves the slot as the previous frame left it rather
// than half rewritten.
pub fn validate(capacity: Capacity, contents: FrameSet.Frame) UpdateError!void {
    if (contents.models.len > capacity.instances) return error.InstanceCapacityExceeded;
    if (contents.joints.len > capacity.joints) return error.JointCapacityExceeded;
    if (contents.previous_joints.len != contents.joints.len) return error.PreviousJointCountMismatch;
    if (contents.lights.len > uniforms.max_lights) return error.LightCapacityExceeded;
    if (contents.shadow_view_projections.len > uniforms.max_shadow_layers)
        return error.ShadowLayerCapacityExceeded;
    for (contents.lights) |light| {
        if (light.shadow_layer == uniforms.no_shadow_layer) continue;
        const layers: u32 = switch (light.kind) {
            .spot => 1,
            .point => 6,
            .directional, .rect => return error.ShadowLayerOutOfRange,
        };
        // As a subtraction, so a layer near the top of u32 cannot wrap into
        // looking in range.
        const available = contents.shadow_view_projections.len;
        if (light.shadow_layer >= available or layers > available - light.shadow_layer)
            return error.ShadowLayerOutOfRange;
    }
}

// Current and previous poses share one descriptor range. The previous pose
// starts at the live count, not at capacity, matching the camera's offset.
pub fn writeJointPoses(storage: []Joint, current: []const Joint, previous: []const Joint) void {
    std.debug.assert(current.len == previous.len);
    std.debug.assert(current.len <= storage.len / 2);
    @memcpy(storage[0..current.len], current);
    @memcpy(storage[current.len..][0..previous.len], previous);
}

pub const FrameSet = struct {
    camera: per_frame.PerFrame(uniforms.Camera),
    instances: per_frame.PerFrame(Instance),
    lights: per_frame.PerFrame(uniforms.Lights),
    joints: per_frame.PerFrame(Joint),
    sun_shadow: per_frame.PerFrame(uniforms.SunShadow),
    sets: Sets,

    pub fn init(
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        allocator: Allocator,
        frames: usize,
        sizes: Capacity,
    ) InitError!FrameSet {
        var camera = try per_frame.PerFrame(uniforms.Camera).init(
            context,
            memory_allocator,
            frames,
            1,
            usageFor(entryFor(.camera).binding.kind),
        );
        errdefer camera.deinit();

        var instances = try per_frame.PerFrame(Instance).init(
            context,
            memory_allocator,
            frames,
            sizes.instances,
            usageFor(entryFor(.instances).binding.kind),
        );
        errdefer instances.deinit();

        // One block per frame, not one light: the array is inside the block.
        var lights = try per_frame.PerFrame(uniforms.Lights).init(
            context,
            memory_allocator,
            frames,
            1,
            usageFor(entryFor(.lights).binding.kind),
        );
        errdefer lights.deinit();

        // One array for the whole frame, not one per skeleton: an instance
        // reaches its own matrices through the base it carries, which is what
        // lets several skeletons be drawn as instances of one draw.
        var joints = try per_frame.PerFrame(Joint).init(
            context,
            memory_allocator,
            frames,
            std.math.mul(usize, sizes.joints, 2) catch return error.SizeOverflow,
            usageFor(entryFor(.joints).binding.kind),
        );
        errdefer joints.deinit();

        // One block per frame, like the camera: the fit it carries is the
        // scene's rather than any one draw's, and both the bake and the lookup
        // read the same copy.
        var sun_shadow = try per_frame.PerFrame(uniforms.SunShadow).init(
            context,
            memory_allocator,
            frames,
            1,
            usageFor(entryFor(.sun_shadow).binding.kind),
        );
        errdefer sun_shadow.deinit();

        var sets = try Sets.init(context, allocator, 1);
        errdefer sets.deinit(context, allocator);

        // Written once. The descriptors name whole buffers; which frame they
        // reach is the dynamic offset's business, and that is per bind.
        //
        // Both infos are named rather than passed inline: the write holds a
        // pointer to each, and it has to outlive the call.
        const infos = [_]vk.DescriptorBufferInfo{
            camera.descriptor(),
            instances.descriptor(),
            lights.descriptor(),
            joints.descriptor(),
            sun_shadow.descriptor(),
        };
        var writes: [entries.len]vk.WriteDescriptorSet = undefined;
        inline for (entries, &writes) |entry, *write| {
            write.* = bufferWrite(
                sets.set(0),
                entry.binding.slot,
                entry.binding.kind,
                &infos[@backingInt(entry.ring)],
            );
        }
        context.device.updateDescriptorSets(&writes, null);

        return .{
            .camera = camera,
            .instances = instances,
            .lights = lights,
            .joints = joints,
            .sun_shadow = sun_shadow,
            .sets = sets,
        };
    }

    pub fn deinit(self: *FrameSet, context: *const Context, allocator: Allocator) void {
        self.sets.deinit(context, allocator);
        self.sun_shadow.deinit();
        self.joints.deinit();
        self.lights.deinit();
        self.instances.deinit();
        self.camera.deinit();
        self.* = undefined;
    }

    // What the rings were built to hold. The scene's joint plan is made against
    // this, so the bound the planner packs into and the bound `validate` tests
    // are the same number rather than two copies of it.
    pub fn capacity(self: *const FrameSet) Capacity {
        return .{ .instances = self.instances.count, .joints = self.joints.count / 2 };
    }

    // What one frame reads. Grouped rather than passed as three parameters
    // because they are written together, under one fence wait.
    pub const Frame = struct {
        // Already in the framebuffer's coordinates. The type is what says so:
        // the ring is the last place the value passes through, and nothing here
        // could tell a flipped block from an unflipped one.
        camera: uniforms.FramebufferCamera,
        models: []const Instance,
        // Every drawn skeleton's matrices, already packed into one run per
        // instance. The bases in `models` index this.
        joints: []const Joint,
        previous_joints: []const Joint = &.{},
        lights: []const uniforms.Light,

        // Defaulted to the off state, which is what a frame that casts no sun
        // shadow carries: it is a supported configuration and not an omission,
        // so it does not have to be written out at every call site that means it.
        sun_shadow: uniforms.SunShadow = .off,

        // The local shadows, off by the same reasoning: the settings default
        // to a strength of zero, and with no matrices no light may name a
        // layer.
        local_shadow: uniforms.LocalShadow = .{},
        shadow_view_projections: []const uniforms.Mat4 = &.{},

        // None by default, for the same reason.
        ambient: uniforms.Ambient = .{},
        // No field by default: the main pass lights from the ambient alone.
        probe_grid: uniforms.ProbeGrid = .none,
        debug: uniforms.Debug = .{},
    };

    // Fill one frame's slot. The caller has waited on that slot's fence; see
    // the invariant on `PerFrame`.
    pub fn update(self: *FrameSet, frame: usize, contents: Frame) UpdateError!void {
        try validate(self.capacity(), contents);

        self.camera.slice(frame)[0] = contents.camera.block;
        @memcpy(self.instances.slice(frame)[0..contents.models.len], contents.models);
        writeJointPoses(self.joints.slice(frame), contents.joints, contents.previous_joints);
        self.lights.slice(frame)[0].fill(contents.lights);
        self.lights.slice(frame)[0].fillShadows(contents.local_shadow, contents.shadow_view_projections);
        self.lights.slice(frame)[0].ambient = contents.ambient;
        self.lights.slice(frame)[0].probe_grid = contents.probe_grid;
        self.lights.slice(frame)[0].debug = contents.debug;
        self.sun_shadow.slice(frame)[0] = contents.sun_shadow;
    }

    // Vulkan specification, vkCmdBindDescriptorSets: dynamic offsets are
    // consumed in increasing binding order. The table above is in that order
    // and names the ring for each slot, so this walks it rather than restating
    // the pairing.
    pub fn dynamicOffsets(self: *const FrameSet, frame: usize) [entries.len]u32 {
        var offsets: [entries.len]u32 = undefined;
        inline for (entries, &offsets) |entry, *offset| {
            offset.* = switch (entry.ring) {
                .camera => self.camera.dynamicOffset(frame),
                .instances => self.instances.dynamicOffset(frame),
                .lights => self.lights.dynamicOffset(frame),
                .joints => self.joints.dynamicOffset(frame),
                .sun_shadow => self.sun_shadow.dynamicOffset(frame),
            };
        }
        return offsets;
    }

    pub fn bind(
        self: *const FrameSet,
        context: *const Context,
        command_buffer: vk.CommandBuffer,
        pipeline_layout: vk.PipelineLayout,
        frame: usize,
    ) void {
        const offsets = self.dynamicOffsets(frame);
        context.device.cmdBindDescriptorSets(
            command_buffer,
            .graphics,
            pipeline_layout,
            0,
            &.{self.sets.set(0)},
            &offsets,
        );
    }

    // The set layout this frame set was built from, which a pipeline layout has
    // to name as its first set. Distinct from the pipeline layout `bind` takes.
    pub fn descriptorSetLayout(self: *const FrameSet) vk.DescriptorSetLayout {
        return self.sets.layout;
    }
};

// Vulkan specification, VkWriteDescriptorSet: the members not selected by
// descriptorType are ignored, but the pointers are not optional in the
// structure, so they are given something valid to point at.
const no_images = [_]vk.DescriptorImageInfo{};
const no_texel_buffers = [_]vk.BufferView{};

fn bufferWrite(
    set: vk.DescriptorSet,
    binding: u32,
    kind: vk.DescriptorType,
    info: *const vk.DescriptorBufferInfo,
) vk.WriteDescriptorSet {
    return .{
        .dst_set = set,
        .dst_binding = binding,
        .dst_array_element = 0,
        .descriptor_count = 1,
        .descriptor_type = kind,
        .p_image_info = &no_images,
        .p_buffer_info = @ptrCast(info),
        .p_texel_buffer_view = &no_texel_buffers,
    };
}
