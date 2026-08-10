const std = @import("std");
const zm = @import("zmath");

// The uniform blocks the frame rings are filled with, and the layout a shader
// created against this module has to declare.
//
// These are the authored side of that agreement rather than a transcription of
// one: nothing here is generated, because generating would move authorship of
// the layout to whichever compiler ran last. Holding a shader against them
// needs that compiler's reflection, which this build does not produce; it is
// the responsibility of whatever supplies the words.
//
// Every block is `extern` so the field order is the declaration order, and
// every one is written into a `PerFrame` ring whose stride the device's offset
// alignment decides.

// The camera every pass reads: the main pass to transform by, the background
// to cast rays along.
pub const Camera = extern struct {
    // Row vectors, so a position multiplies from the left. zmath stores a `Mat`
    // as four rows and the shader reads it the same way; nothing transposes.
    view_projection: zm.Mat align(16),

    // xyz is the eye in world space. The fourth lane is padding that the
    // sixteen-byte alignment would insert anyway, so it is named rather than
    // left to the compiler.
    position: [4]f32 align(16),

    // The view ray a pass covering the target reconstructs, in the shape
    // `lenore-scene`'s `Camera.rayBasis` produces: for a device coordinate
    // (x, y) the ray leaving the eye is `ray_front + ray_right * x + ray_up * y`.
    // Affine in that coordinate, so it interpolates exactly across a triangle
    // covering the whole target and no shader stage divides.
    //
    // In device coordinates, like `view_projection` above: `vulkanClipCamera`
    // negates `ray_up` where it negates the matrix's second column, because +1
    // is the last row of the framebuffer and the top of the scene's clip space.
    //
    // The fourth lane of each is the alignment's padding, named for the same
    // reason `position`'s is.
    ray_right: [4]f32 align(16),
    ray_up: [4]f32 align(16),
    ray_front: [4]f32 align(16),
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

// One element of the light block below.
//
// Flat rather than a union, because a shader reads a fixed layout and the loop
// over lights branches on `kind`. Each kind still has its own fields: a
// directional light has no position and no range, and the lanes it does not use
// are zero rather than reinterpreted. The reference packs a direction into the
// position lanes and guards both accessors with an assert on the tag; here
// there is nothing to guard.
//
// The offsets are std140's, which is what a Slang constant buffer lays a struct
// out as. Nothing here restates them: `tests/shader_reflection.zig` holds every field
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

// The light block, read by every shaded fragment.
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

// The sun's fit, read by the bake and by the lookup through the same binding:
// both transform by one matrix, so they take it from one block.
//
// Not lanes added to the light it shadows. The bake binds no light array at all,
// and a scene with several directional lights has at most one map, so folding
// these fields into `Light` would put them on every entry to be read on one.
pub const SunShadow = extern struct {
    // World space to the sun's clip space, from `SunShadowFit` in lenore-scene.
    //
    // `vulkanClip` is not applied to it, unlike the camera's. The bake
    // rasterizes through this matrix and the lookup turns its own product with
    // it into texture coordinates, so a flip on one side has to be matched on
    // the other; leaving it out of both is what makes the two cancel.
    view_projection: zm.Mat align(16),

    // How much of the sun's direct term a shadowed surface loses, in [0, 1],
    // which `SunShadowSettings.clampedStrength` is what holds it to. Zero is the
    // off state and the shader tests it rather than a separate flag: a disabled
    // shadow and a shadow of no strength are the same picture.
    strength: f32,

    // How far the lookup point is pushed off the surface along its normal, in
    // world units. `SunShadowSettings.normalOffsetWorld` converts the authored
    // texel count through the fit's own texel size, so it follows scene scale
    // and map resolution without being retuned.
    normal_offset: f32,

    // Which entry of the light block this map was baked for. A fit is built from
    // one direction, so exactly one light can be shadowed through it, and naming
    // the index here is what keeps the pairing from being an ordering convention
    // that the host and the shader each state separately.
    light: u32,

    // Reaching the sixteen-byte multiple the block is strided to. Named rather
    // than left to the compiler so the shader's own padding field has something
    // to match.
    padding: u32 = 0,

    // A frame that casts no sun shadow. The matrix is never read at this
    // strength, on either side: the lookup returns before transforming anything,
    // and a bake is not recorded at all.
    pub const off: SunShadow = .{
        .view_projection = zm.identity(),
        .strength = 0,
        .normal_offset = 0,
        .light = 0,
    };
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

// A camera block in the coordinates the framebuffer counts in, which is the
// only kind a frame may be filled with.
//
// A distinct type rather than a convention, because the two are the same fields
// and differ only in a sign that nothing but a picture can check. `Camera` alone
// left "flipped" and "not flipped" indistinguishable at every call site, and a
// path that flipped the matrix while leaving the ray basis alone type-checked,
// passed every test, and rendered the background upside down.
//
// `vulkanClipCamera` is the only thing that produces one.
pub const FramebufferCamera = struct {
    block: Camera,
};

// The whole camera block in the coordinates the framebuffer counts in.
//
// Both halves of the flip in one call because they describe one axis. A shader
// that transforms a vertex by the matrix and a shader that reconstructs a ray
// from the basis have to agree about which way device y points, and applying
// one of the two is a background upside down against the scene in front of it.
//
// The padding lane of `ray_up` is left alone. It is not read on either side, and
// negating a zero is a difference between two values that mean the same thing.
pub fn vulkanClipCamera(camera: Camera) FramebufferCamera {
    var flipped = camera;
    flipped.view_projection = vulkanClip(camera.view_projection);
    inline for (0..3) |axis| flipped.ray_up[axis] = -camera.ray_up[axis];
    return .{ .block = flipped };
}
