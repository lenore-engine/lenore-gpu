const std = @import("std");
const zm = @import("zmath");

// The uniform blocks the shaders declare, mirrored on this side.
//
// Written by hand and checked against the compiler's reflection output rather
// than generated from it. Generating would move authorship of the layout to the
// tool; checking keeps it here and still fails the build when the two disagree.
// The check is `tests/reflection.zig`, which reads the JSON the build emits
// beside each SPIR-V module.
//
// Every block is `extern` so the field order is the declaration order, and
// every one is written into a `PerFrame` ring whose stride the device's offset
// alignment decides.

// Mirrors `CameraUbo` in scene.slang.
pub const Camera = extern struct {
    // Row vectors, so a position multiplies from the left. zmath stores a `Mat`
    // as four rows and the shader reads it the same way; nothing transposes.
    view_projection: zm.Mat align(16),

    // xyz is the eye in world space. The fourth lane is padding that the
    // sixteen-byte alignment would insert anyway, so it is named rather than
    // left to the compiler.
    position: [4]f32 align(16),
};

// How many lights one frame's block holds.
//
// Provisional, and derived from what assets ask for rather than from a device
// limit: across the Khronos sample corpus the most any one file references is
// eight, in `PointLightIntensityTest`. Sixteen is twice that, and it keeps the
// block at 1040 bytes, an order below the 16384 every implementation has to
// offer for a uniform buffer range (vk.xml, maxUniformBufferRange).
//
// What moves it is a scene that overflows it, which the count check in
// `FrameSet.update` reports rather than truncates.
pub const max_lights = 16;

// Mirrors `Light` in scene.slang, one element of the block below.
//
// Flat rather than a union, because a shader reads a fixed layout and the loop
// over lights branches on `kind`. Each kind still has its own fields: a
// directional light has no position and no range, and the lanes it does not use
// are zero rather than reinterpreted. The reference packs a direction into the
// position lanes and guards both accessors with an assert on the tag; here
// there is nothing to guard.
//
// The offsets are std140's, which is what a Slang constant buffer lays a struct
// out as. Nothing here restates them: `tests/reflection.zig` holds every field
// against the compiler's own account of the shader.
pub const Light = extern struct {
    pub const Kind = enum(u32) { directional, point, spot };

    // World space, for `point` and `spot`. Unused by `directional`.
    position: [3]f32 align(16),
    kind: Kind,

    // Linear, not premultiplied by the intensity.
    colour: [3]f32,
    intensity: f32,

    // Unit, and pointing the way the light travels, which is the convention
    // `lenore-scene`'s `Light` carries and normalizes. Unused by `point`.
    direction: [3]f32,

    // Distance past which the light contributes nothing. Unused by
    // `directional`.
    range: f32,

    // The spot cone, as cosines against the axis. `cos_inner` is the larger:
    // cosine falls as the angle opens. Both zero for the other two kinds.
    cos_inner: f32,
    cos_outer: f32,

    // Reaching the sixteen-byte multiple an array of these is strided by. Named
    // rather than left to the compiler so the shader's own padding field has
    // something to match.
    padding: [2]f32 = .{ 0, 0 },

    pub fn directional(colour: [3]f32, intensity: f32, direction: [3]f32) Light {
        return .{
            .position = .{ 0, 0, 0 },
            .kind = .directional,
            .colour = colour,
            .intensity = intensity,
            .direction = direction,
            .range = 0,
            .cos_inner = 0,
            .cos_outer = 0,
        };
    }

    pub fn point(colour: [3]f32, intensity: f32, position: [3]f32, range: f32) Light {
        return .{
            .position = position,
            .kind = .point,
            .colour = colour,
            .intensity = intensity,
            .direction = .{ 0, 0, 0 },
            .range = range,
            .cos_inner = 0,
            .cos_outer = 0,
        };
    }

    pub const Spot = struct {
        position: [3]f32,
        direction: [3]f32,
        range: f32,
        cos_inner: f32,
        cos_outer: f32,
    };

    pub fn spot(colour: [3]f32, intensity: f32, cone: Spot) Light {
        return .{
            .position = cone.position,
            .kind = .spot,
            .colour = colour,
            .intensity = intensity,
            .direction = cone.direction,
            .range = cone.range,
            .cos_inner = cone.cos_inner,
            .cos_outer = cone.cos_outer,
        };
    }
};

// Mirrors `LightsUbo` in scene.slang.
//
// A fixed array rather than a storage buffer sized to the scene. The block is
// read by every fragment, so it wants the constant path; the array's length is
// what the shader's loop is bounded by, and `count` is how much of it holds a
// light. The lanes past `count` are not written, so nothing may read them.
pub const Lights = extern struct {
    // x is how many entries of `lights` are live. The other three lanes are the
    // alignment the array needs either way.
    count: [4]u32 align(16),
    lights: [max_lights]Light,

    // Write a frame's lights into the block.
    //
    // Total, because the length is bounded before anything is written:
    // `frame_set.validate` rejects a frame with more than `max_lights` in it,
    // and `FrameSet.update` calls it before reaching here.
    //
    // The entries past the count are left as they were. The shader's loop stops
    // at the count, so they are not read, and clearing them would be
    // `max_lights` writes a frame for nothing.
    pub fn fill(self: *Lights, lights: []const Light) void {
        self.count = .{ @intCast(lights.len), 0, 0, 0 };
        @memcpy(self.lights[0..lights.len], lights);
    }
};

// The camera's clip space with the Vulkan Y flip applied.
//
// `lenore-scene` produces clip space with Y up; a framebuffer counts rows
// downward. The flip is this side's, and it goes in the matrix rather than in
// the viewport because the matrix is what a shader inverts: reconstructing a
// world position from a framebuffer coordinate is `inverse(view_projection)`
// times that coordinate, and an inverse that leaves the flip out is wrong in
// one axis with nothing on screen to show it.
//
// The whole second column is negated, not just its diagonal entry. Negating a
// projection's `[1][1]` alone is the same thing only while the rest of that
// column is zero, which is true of a bare perspective matrix and not of the
// product with a view.
pub fn vulkanClip(view_projection: zm.Mat) zm.Mat {
    var flipped = view_projection;
    inline for (0..4) |row| flipped[row][1] = -flipped[row][1];
    return flipped;
}
