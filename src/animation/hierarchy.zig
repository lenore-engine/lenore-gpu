const std = @import("std");

// A slot is a node of a hierarchy that is animated as one array: a skeleton's
// bones, or the dynamic nodes of a document. Both are stored in topological
// order, so a parent always precedes its children and one forward pass computes
// every world transform.
pub const Slot = u16;

// The parent of a root. A dense array of Slot rather than of optionals, so the
// array stays plain integers and stays smaller: the assert below measures by how
// much rather than claiming it.
//
// The value is only safe as a sentinel because no valid slot can equal it, and
// that is established by construction rather than asserted: max_slots is derived
// from it, so widening Slot moves both together and the two cannot drift apart.
pub const no_parent: Slot = std.math.maxInt(Slot);
pub const max_slots: usize = no_parent;

comptime {
    // Slots run from zero to max_slots - 1, which leaves no_parent outside the
    // range of every index a hierarchy can hold.
    std.debug.assert(max_slots - 1 < no_parent);
    std.debug.assert(@sizeOf(Slot) * 2 == @sizeOf(?Slot));
}

pub const TopologyError = error{
    ParentNotBeforeChild,
    TooManySlots,
};

// The one property both animated hierarchies need: a parent is either absent or
// an earlier slot. Checking it once here is what lets their per-frame loops read
// a parent's transform with no bounds check and no ordering test.
pub fn validate(parents: []const Slot) TopologyError!void {
    if (parents.len > max_slots) return error.TooManySlots;
    for (parents, 0..) |parent, index| {
        if (parent == no_parent) continue;
        if (parent >= index) return error.ParentNotBeforeChild;
    }
}
