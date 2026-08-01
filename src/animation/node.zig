const std = @import("std");
const zm = @import("zmath");
const clip_module = @import("clip.zig");
const hierarchy = @import("hierarchy.zig");
const composeTransform = @import("../trs.zig").composeTransform;

const Allocator = std.mem.Allocator;
const Animation = clip_module.Animation;

// Rigid animation: channels that move whole nodes, such as a door, a lift or a
// spinning prop, rather than the joints of a skin.
//
// The split is the same one the skeleton makes. A template holds what every
// instance shares and is never written after load; an animator holds one
// instance's playback state. Two doors from one asset share the parsed clips and
// the hierarchy, and differ only in where they are in their clip.
//
// Both are laid out as arrays in topological order, so propagating world
// transforms is one linear pass rather than a recursive walk.
pub const Template = struct {
    // Parent slot of each slot, or hierarchy.no_parent for the root of a
    // dynamic subtree. Always less than its own index, verified once at
    // construction.
    parent: []hierarchy.Slot,
    // World transform of the static ancestors above each subtree root, and
    // identity elsewhere. Held for every slot rather than for roots only,
    // because that makes the propagation below pick between prefix and parent
    // by the same index instead of by an indirection.
    prefix: []zm.Mat,
    bind_translations: []zm.Vec,
    bind_rotations: []zm.Quat,
    bind_scales: []zm.Vec,
    // The bind local matrix per slot, composed at load. Kept beside the bind
    // transform rather than derived from it. glTF 2.0 specification, 5.25: a
    // node targeted by an animation channel may carry only translation,
    // rotation and scale, and matrix must not be present. Only such a node.
    // One that merely inherits motion may carry a matrix instead, and composing
    // the three would silently drop it.
    bind_local: []zm.Mat,
    // Every clip in the document, with channels already addressed by slot.
    clips: []Animation,
    // Per clip, the slots at least one channel targets, deduplicated. Slots
    // that only inherit motion, which is most of them in a typical asset, keep
    // their bind local matrix and are never recomposed.
    animated_slots: [][]u16,

    pub const InitError = error{
        AnimatedSlotOutOfRange,
        ClipTargetOutOfRange,
        SlotCountMismatch,
    } || hierarchy.TopologyError;

    // Takes ownership of every array passed in, including the clips, from this
    // call and including on failure: a rejected document is destroyed here
    // rather than handed back for a caller to take apart field by field. That is
    // the same discipline the owning storage and the reference cache follow.
    //
    // It validates rather than copying. This data is assembled by a loader that
    // has just allocated it, and duplicating a whole document's animation to
    // check it would be the largest copy in the load path.
    //
    // The checks are what let the per-frame loops below index without one.
    pub fn init(allocator: Allocator, owned: Template) InitError!Template {
        var taken = owned;
        errdefer taken.deinit(allocator);

        const slot_count = owned.parent.len;
        try hierarchy.validate(owned.parent);
        if (owned.prefix.len != slot_count or
            owned.bind_translations.len != slot_count or
            owned.bind_rotations.len != slot_count or
            owned.bind_scales.len != slot_count or
            owned.bind_local.len != slot_count)
            return error.SlotCountMismatch;
        if (owned.animated_slots.len != owned.clips.len) return error.SlotCountMismatch;

        for (owned.animated_slots) |slots| {
            for (slots) |slot| {
                if (slot >= slot_count) return error.AnimatedSlotOutOfRange;
            }
        }
        // A channel addressing a slot this hierarchy does not have would make
        // sampling fail every frame, silently, once playback reached it.
        for (owned.clips) |*clip| {
            if (clip.slot_count > slot_count) return error.ClipTargetOutOfRange;
        }
        return taken;
    }

    pub fn deinit(self: *Template, allocator: Allocator) void {
        allocator.free(self.parent);
        allocator.free(self.prefix);
        allocator.free(self.bind_translations);
        allocator.free(self.bind_rotations);
        allocator.free(self.bind_scales);
        allocator.free(self.bind_local);
        for (self.clips) |*clip| clip.deinit(allocator);
        allocator.free(self.clips);
        for (self.animated_slots) |slots| allocator.free(slots);
        allocator.free(self.animated_slots);
        self.* = undefined;
    }

    pub fn slotCount(self: *const Template) usize {
        return self.parent.len;
    }
};

pub const PlayError = error{NoSuchClip};

// One instance's playback. Every array is allocated in init and none resizes,
// so update allocates nothing.
pub const Animator = struct {
    template: *const Template,
    // Elapsed since the active clip started, not a position inside it.
    elapsed: f32,
    // Null means nothing is playing: update returns immediately and an idle
    // instance costs nothing per frame.
    active_clip: ?u16,

    local_translations: []zm.Vec,
    local_rotations: []zm.Quat,
    local_scales: []zm.Vec,
    // Composed locals. Only the active clip's animated slots are rewritten each
    // frame; the rest keep the bind local they were reset to.
    local_transforms: []zm.Mat,
    world_transforms: []zm.Mat,

    pub fn init(allocator: Allocator, template: *const Template) Allocator.Error!Animator {
        const count = template.slotCount();

        const local_translations = try allocator.alloc(zm.Vec, count);
        errdefer allocator.free(local_translations);
        const local_rotations = try allocator.alloc(zm.Quat, count);
        errdefer allocator.free(local_rotations);
        const local_scales = try allocator.alloc(zm.Vec, count);
        errdefer allocator.free(local_scales);
        const local_transforms = try allocator.alloc(zm.Mat, count);
        errdefer allocator.free(local_transforms);
        const world_transforms = try allocator.alloc(zm.Mat, count);

        var animator: Animator = .{
            .template = template,
            .elapsed = 0.0,
            .active_clip = null,
            .local_translations = local_translations,
            .local_rotations = local_rotations,
            .local_scales = local_scales,
            .local_transforms = local_transforms,
            .world_transforms = world_transforms,
        };

        // World transforms have to be valid for rendering even if no clip is
        // ever played, so the bind pose is propagated once here.
        animator.resetToBind();
        animator.propagate();
        return animator;
    }

    pub fn deinit(self: *Animator, allocator: Allocator) void {
        allocator.free(self.local_translations);
        allocator.free(self.local_rotations);
        allocator.free(self.local_scales);
        allocator.free(self.local_transforms);
        allocator.free(self.world_transforms);
        self.* = undefined;
    }

    // Restarts a clip from its beginning. The reset is what keeps a switch
    // between clips deterministic: without it, slots the new clip does not
    // target would keep whatever the previous one left in them.
    pub fn play(self: *Animator, clip_index: u16) PlayError!void {
        if (clip_index >= self.template.clips.len) return error.NoSuchClip;
        self.resetToBind();
        self.active_clip = clip_index;
        self.elapsed = 0.0;
    }

    // Freezes the current pose. The last propagated world transforms stay valid.
    pub fn stop(self: *Animator) void {
        self.active_clip = null;
    }

    // Advances time, samples the active clip, recomposes the slots it touches
    // and propagates. Host work only: nothing here talks to a device.
    pub fn update(self: *Animator, delta_time: f32) void {
        const clip_index = self.active_clip orelse return;
        const clip = &self.template.clips[clip_index];

        self.elapsed += delta_time;
        // Wrapping here bounds the accumulator rather than deciding playback:
        // cursorAt below maps whatever it receives onto the clip's window, and
        // an f32 that grows without bound loses the precision this needs.
        const span = clip.loopSpan();
        if (span > 0.0 and self.elapsed >= span) self.elapsed = @mod(self.elapsed, span);

        // The template rejected any clip whose channels reach past this
        // hierarchy, so the capacity this reports on cannot be short. Stopping
        // rather than continuing keeps a mistake in that reasoning visible as a
        // frozen prop instead of as a wrong pose.
        clip.sample(
            clip.cursorAt(self.elapsed),
            self.local_translations,
            self.local_rotations,
            self.local_scales,
        ) catch return;

        // Only the slots a channel targets are recomposed. In a typical asset
        // most dynamic slots merely inherit motion, and recomposing all of them
        // every frame would be work with no effect.
        for (self.template.animated_slots[clip_index]) |slot| {
            self.local_transforms[slot] = composeTransform(
                self.local_translations[slot],
                self.local_rotations[slot],
                self.local_scales[slot],
            );
        }

        self.propagate();
    }

    fn resetToBind(self: *Animator) void {
        const template = self.template;
        @memcpy(self.local_translations, template.bind_translations);
        @memcpy(self.local_rotations, template.bind_rotations);
        @memcpy(self.local_scales, template.bind_scales);
        // The bind local matrix rather than a composition of the bind
        // transform, because a node carrying a matrix property has no
        // meaningful translation, rotation and scale to compose.
        @memcpy(self.local_transforms, template.bind_local);
    }

    // One forward pass. Topological order guarantees a parent's world transform
    // is final before a child reads it, so the sequential scan is the hierarchy
    // traversal, and the template's validation is what makes the parent index
    // safe to use without a check.
    fn propagate(self: *Animator) void {
        for (
            self.local_transforms,
            self.world_transforms,
            self.template.parent,
            self.template.prefix,
        ) |local, *world, parent, prefix| {
            world.* = zm.mul(local, if (parent == hierarchy.no_parent)
                prefix
            else
                self.world_transforms[parent]);
        }
    }
};
