const std = @import("std");
const zm = @import("zmath");
const hierarchy = @import("hierarchy.zig");
const composeTransform = @import("../trs.zig").composeTransform;

const Allocator = std.mem.Allocator;

// Two index spaces meet here and are not interchangeable.
//
// A slot is a node of the skeleton's own hierarchy, ordered so that a parent
// always precedes its children. That ordering is what lets one forward pass
// compute every world transform.
//
// A joint is an entry of the skin, in the order the asset declared, which is the
// index space the vertex joint attribute addresses. joint_slot maps one to the
// other, and the two counts differ whenever the hierarchy carries nodes that no
// vertex is bound to.
pub const Template = struct {
    // Parent slot of each slot, or hierarchy.no_parent for a root. Always less
    // than its own index, which init verifies once so the pose pass never has
    // to.
    slot_parent: []hierarchy.Slot,
    // Transform above each root, folded in at construction so a root's world
    // transform is one multiplication rather than a walk out of the skeleton.
    slot_prefix: []zm.Mat,
    bind_translations: []zm.Vec,
    bind_rotations: []zm.Quat,
    bind_scales: []zm.Vec,
    // Inverse bind matrix per joint, from the asset.
    inverse_bind: []zm.Mat,
    // Slot each joint reads its world transform from.
    joint_slot: []u16,

    pub const Init = struct {
        slot_parent: []const hierarchy.Slot,
        slot_prefix: []const zm.Mat,
        bind_translations: []const zm.Vec,
        bind_rotations: []const zm.Quat,
        bind_scales: []const zm.Vec,
        inverse_bind: []const zm.Mat,
        joint_slot: []const u16,
    };

    pub const InitError = error{
        JointSlotOutOfRange,
        SlotCountMismatch,
    } || hierarchy.TopologyError || Allocator.Error;

    // Every check here runs once, on data an asset supplied, and each one buys a
    // per-frame path that needs no check at all. They are errors rather than
    // asserts because the shipping build removes asserts: an unsorted parent
    // then reads a world transform this frame has not written, and an
    // out-of-range one reads outside the array.
    pub fn init(allocator: Allocator, params: Init) InitError!Template {
        const slot_count = params.slot_parent.len;
        try hierarchy.validate(params.slot_parent);
        if (params.slot_prefix.len != slot_count or
            params.bind_translations.len != slot_count or
            params.bind_rotations.len != slot_count or
            params.bind_scales.len != slot_count)
            return error.SlotCountMismatch;
        if (params.joint_slot.len != params.inverse_bind.len)
            return error.SlotCountMismatch;

        for (params.joint_slot) |slot| {
            if (slot >= slot_count) return error.JointSlotOutOfRange;
        }

        const slot_parent = try allocator.dupe(hierarchy.Slot, params.slot_parent);
        errdefer allocator.free(slot_parent);
        const slot_prefix = try allocator.dupe(zm.Mat, params.slot_prefix);
        errdefer allocator.free(slot_prefix);
        const bind_translations = try allocator.dupe(zm.Vec, params.bind_translations);
        errdefer allocator.free(bind_translations);
        const bind_rotations = try allocator.dupe(zm.Quat, params.bind_rotations);
        errdefer allocator.free(bind_rotations);
        const bind_scales = try allocator.dupe(zm.Vec, params.bind_scales);
        errdefer allocator.free(bind_scales);
        const inverse_bind = try allocator.dupe(zm.Mat, params.inverse_bind);
        errdefer allocator.free(inverse_bind);
        const joint_slot = try allocator.dupe(u16, params.joint_slot);

        return .{
            .slot_parent = slot_parent,
            .slot_prefix = slot_prefix,
            .bind_translations = bind_translations,
            .bind_rotations = bind_rotations,
            .bind_scales = bind_scales,
            .inverse_bind = inverse_bind,
            .joint_slot = joint_slot,
        };
    }

    pub fn deinit(self: *Template, allocator: Allocator) void {
        allocator.free(self.slot_parent);
        allocator.free(self.slot_prefix);
        allocator.free(self.bind_translations);
        allocator.free(self.bind_rotations);
        allocator.free(self.bind_scales);
        allocator.free(self.inverse_bind);
        allocator.free(self.joint_slot);
        self.* = undefined;
    }

    pub fn slotCount(self: *const Template) usize {
        return self.slot_parent.len;
    }

    pub fn jointCount(self: *const Template) usize {
        return self.inverse_bind.len;
    }
};

// One instance's pose. The template is borrowed and outlives this.
//
// Local transforms are what an animation writes; world and final transforms are
// what pose derives from them. Keeping them apart is what lets several instances
// share one skeleton and animate independently.
pub const Pose = struct {
    template: *const Template,
    local_translations: []zm.Vec,
    local_rotations: []zm.Quat,
    local_scales: []zm.Vec,
    world_transforms: []zm.Mat,
    // One per joint, in the vertex attribute's index space. This is what a
    // skinning shader reads, and it is uploaded as it stands: no transpose on
    // either side.
    //
    // That works because two conventions cancel. zmath/src/root.zig composes for
    // v * M and declares Mat as [4]F32x4, four rows stored in order. The OpenGL
    // Shading Language, section Uniform and Shader Storage Block Layout
    // Qualifiers, reads a matrix column by column unless a block declares
    // row_major, so a shader receives the transpose of what was written. It then
    // applies M * v, which computes the v * M the composition intended.
    //
    // Inserting a transpose anywhere in that chain breaks it, which is a mesh
    // that comes apart under animation rather than anything that reports an
    // error.
    joint_transforms: []zm.Mat,

    pub fn init(allocator: Allocator, template: *const Template) Allocator.Error!Pose {
        const local_translations = try allocator.dupe(zm.Vec, template.bind_translations);
        errdefer allocator.free(local_translations);
        const local_rotations = try allocator.dupe(zm.Quat, template.bind_rotations);
        errdefer allocator.free(local_rotations);
        const local_scales = try allocator.dupe(zm.Vec, template.bind_scales);
        errdefer allocator.free(local_scales);

        const world_transforms = try allocator.alloc(zm.Mat, template.slotCount());
        errdefer allocator.free(world_transforms);
        @memset(world_transforms, zm.identity());

        const joint_transforms = try allocator.alloc(zm.Mat, template.jointCount());
        @memset(joint_transforms, zm.identity());

        return .{
            .template = template,
            .local_translations = local_translations,
            .local_rotations = local_rotations,
            .local_scales = local_scales,
            .world_transforms = world_transforms,
            .joint_transforms = joint_transforms,
        };
    }

    pub fn deinit(self: *Pose, allocator: Allocator) void {
        allocator.free(self.local_translations);
        allocator.free(self.local_rotations);
        allocator.free(self.local_scales);
        allocator.free(self.world_transforms);
        allocator.free(self.joint_transforms);
        self.* = undefined;
    }

    pub fn jointCount(self: *const Pose) usize {
        return self.joint_transforms.len;
    }

    // One forward pass over the slots, then one over the joints. The slot order
    // guarantees a parent's world transform is already written when a child
    // reads it, and the template verified that order at construction, so this
    // runs with no branch beyond the root test and no bounds check to justify.
    pub fn evaluate(self: *Pose) void {
        const template = self.template;
        for (template.slot_parent, 0..) |parent, index| {
            const local = composeTransform(
                self.local_translations[index],
                self.local_rotations[index],
                self.local_scales[index],
            );
            self.world_transforms[index] = if (parent == hierarchy.no_parent)
                zm.mul(local, template.slot_prefix[index])
            else
                zm.mul(local, self.world_transforms[parent]);
        }

        for (
            self.joint_transforms,
            template.inverse_bind,
            template.joint_slot,
        ) |*joint, inverse_bind, slot| {
            joint.* = zm.mul(inverse_bind, self.world_transforms[slot]);
        }
    }

    pub fn resetToBindPose(self: *Pose) void {
        @memcpy(self.local_translations, self.template.bind_translations);
        @memcpy(self.local_rotations, self.template.bind_rotations);
        @memcpy(self.local_scales, self.template.bind_scales);
    }

    // Displaces the first slot, which is where a root motion track applies. A
    // skeleton with no slots has nothing to displace.
    pub fn applyRootMotion(self: *Pose, offset: zm.Vec) void {
        if (self.local_translations.len == 0) return;
        self.local_translations[0] = self.local_translations[0] + offset;
    }
};
