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

// A matrix as the device reads it: four rows of four floats, the order zmath
// keeps a `Mat` in, so nothing transposes.
//
// An array rather than `zm.Mat`, because the device reads these bytes and Zig
// defines no memory layout for `@Vector`: an extern struct may not hold one,
// and comptime refuses to read one through a float pointer. Arrays of `f32`
// have the layout the shader's mirror is held against.
pub const Mat4 = [4][4]f32;

// By value: each row coerces from a vector to an array, which copies the four
// lanes and reinterprets no memory.
pub fn mat4(m: zm.Mat) Mat4 {
    return .{ m[0], m[1], m[2], m[3] };
}

// The camera every pass reads: the main pass to transform by, the background
// to cast rays along.
pub const Camera = extern struct {
    // Row vectors, so a position multiplies from the left. zmath stores a `Mat`
    // as four rows and the shader reads it the same way; nothing transposes.
    view_projection: Mat4 align(16),

    // xyz is the eye in world space, and w is the frame's clock in seconds.
    //
    // The clock rides in a lane the sixteen-byte alignment reserves anyway, so
    // it costs nothing and moves nothing. It belongs in this block rather than
    // beside it for the reason Godot gives for keeping its own `time` in the
    // scene data block: this is where a frame's view-independent globals live,
    // and a shader that animates anything needs one exactly as much as it needs
    // the eye (`servers/rendering/renderer_rd/shaders/scene_data_inc.glsl`,
    // `struct SceneData`).
    //
    // It is folded into an hour, so anything animated on it must be periodic in
    // that hour. `lenore`'s `FrameTime.shaderSeconds` is what folds it and says
    // why.
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

    // Last frame's `view_projection`, which the depth prepass projects each
    // vertex's last position through to write how far it moved on screen. The
    // engine passes the current matrix again where there is no last frame to
    // move from: the first one, and the first after a resize.
    previous_view_projection: Mat4 align(16),

    // x is where last frame's joint matrices begin in the joint array: the
    // engine packs the previous pose after the current one, in the same
    // layout, so a skinned vertex finds both by the same base. The other lanes
    // are the alignment's padding.
    motion: [4]u32 align(16) = .{ 0, 0, 0, 0 },

    // xy is the sub-pixel offset `view_projection` was moved by this frame, in
    // the framebuffer's device coordinates (`jitterClip`); zero when the frame
    // is not jittered. A reader that wants where a point really is, rather than
    // where this frame's samples put it, takes it back out: the prepass's
    // motion, and the occlusion pass's reconstruction. `previous_view_projection`
    // is never jittered. The other lanes are the alignment's padding.
    jitter: [4]f32 align(16) = .{ 0, 0, 0, 0 },
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

// How many local shadow layers one frame's block holds matrices for: a spot
// takes one, a point light six.
//
// DECIDE: provisional. Twenty-four is four shadowed point lights, or any mix
// down to twenty-four spots, and keeps the matrices at 1536 bytes of a block
// still far under the 16384 every implementation offers (vk.xml,
// maxUniformBufferRange). Closed by what Reclaim's base needs lit at once.
pub const max_shadow_layers = 24;

// The value of `Light.shadow_layer` for a light that casts no shadow.
pub const no_shadow_layer: u32 = std.math.maxInt(u32);

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
    pub const Kind = enum(u32) { directional, point, spot, rect };

    // World space. The centre for `rect`, the source for `point` and `spot`.
    // Unused by `directional`.
    position: [3]f32 align(16),
    kind: Kind,

    // Linear, not premultiplied by the intensity.
    colour: [3]f32,
    intensity: f32,

    // Unit, and pointing the way the light travels, which is the convention
    // `lenore-scene`'s `Light` carries and normalizes. For `rect` it is the face
    // the rectangle emits from, so a surface behind it receives nothing. Unused
    // by `point`.
    direction: [3]f32,

    // Distance past which the light contributes nothing. Unused by
    // `directional`.
    range: f32,

    // The spot cone, as cosines against the axis. `cos_inner` is the larger:
    // cosine falls as the angle opens. Both zero for the other kinds.
    cos_inner: f32,
    cos_outer: f32,

    // The light's first layer in the local shadow array, or `no_shadow_layer`.
    // A spot reads that layer; a point light reads it and the five after it,
    // in the face order of `LocalShadowFit` in lenore-scene. Every other kind
    // carries `no_shadow_layer`, and `frame_set.validate` refuses one that
    // does not, or a run that ends past the frame's matrices.
    shadow_layer: u32 = no_shadow_layer,

    // `LocalShadowFit.texel_slope`: world units across one texel at unit
    // distance from the light, which the lookup scales by the fragment's
    // distance to turn the authored normal offset into a world one. Zero for a
    // light with no layer.
    shadow_texel_slope: f32 = 0,

    // The rectangle, for `rect`, and zero for every other kind.
    //
    // `half_edge` is half of one edge, as a vector from the centre at
    // `position`. The other edge is perpendicular to it and to `direction`, with
    // `half_extent` for its half length, so the four corners are the centre plus
    // and minus `half_edge`, plus and minus `normalize(cross(direction,
    // half_edge))` scaled by `half_extent`.
    //
    // One vector and one scalar rather than two vectors. The second edge is
    // fixed by the first and the emitting face up to its length, so storing it
    // as well would allow a rectangle whose own edges disagree with the
    // direction it claims to emit in, and nothing downstream could tell which of
    // the three was meant.
    half_edge: [3]f32 = .{ 0, 0, 0 },
    half_extent: f32 = 0,

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

    pub const Rect = struct {
        centre: [3]f32,

        // Unit, and the face the rectangle emits from.
        direction: [3]f32,

        // Half of one edge, as a vector from the centre, and the half length of
        // the edge perpendicular to it. See the fields for the geometry these
        // describe.
        half_edge: [3]f32,
        half_extent: f32,

        range: f32,
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

    pub fn rect(colour: [3]f32, intensity: f32, panel: Rect) Light {
        return .{
            .position = panel.centre,
            .kind = .rect,
            .colour = colour,
            .intensity = intensity,
            .direction = panel.direction,
            .range = panel.range,
            .cos_inner = 0,
            .cos_outer = 0,
            .half_edge = panel.half_edge,
            .half_extent = panel.half_extent,
        };
    }
};

// How the local shadows are read, for every light that has layers. One pair
// for the scene rather than lanes on each light, like the sun's: they are look
// controls, and nothing about one lamp makes its shadow want a different one.
pub const LocalShadow = extern struct {
    // What a shadowed surface loses of a local light's direct term, in [0, 1].
    // Zero switches the lookup off for every light, as the sun's strength does.
    strength: f32 = 0,

    // How far the lookup point is pushed off the surface along its normal, in
    // texels of the light's layer at the fragment's distance.
    normal_offset_texels: f32 = 1,

    // The radius of every local light's emitting surface, in world units. A
    // shadow's edge softens with it, more the farther the surface it falls
    // on is behind the caster. Zero is a point and the hardest edge the map
    // can give. One value for every light: `Light` has no lane to spare.
    source_radius: f32 = 0,

    // The near plane of every layer, `LocalShadowFit`'s, which the lookup
    // needs to turn a stored depth into a distance. The engine writes it; a
    // caller's value is replaced.
    near: f32 = 0,
};

// Light from everywhere at once: a sky radiance from above and a ground
// radiance from below, blended by direction. What a scene with no prefiltered
// environment has instead of one, and what an interior adds to the one it has.
// Linear, and in the same units as the environment's maps; zero is none, the
// default, so a scene that sets nothing renders as before.
pub const Ambient = extern struct {
    sky: [3]f32 align(16) = .{ 0, 0, 0 },
    padding_0: f32 = 0,
    ground: [3]f32 = .{ 0, 0, 0 },
    padding_1: f32 = 0,
};

// Where the indirect-light probes stand: probe (0, 0, 0) at `origin`, the
// rest `spacing` apart, `counts` per axis, X fastest. Mirrors
// `ProbeGridParams` in include/probe_field.slang, which the main pass and the
// probe update both read. `enabled` zero is no field.
pub const ProbeGrid = extern struct {
    origin: [3]f32 align(16),
    spacing: f32,
    counts: [3]u32,
    enabled: u32,

    pub const none: ProbeGrid = .{ .origin = .{ 0, 0, 0 }, .spacing = 1, .counts = .{ 0, 0, 0 }, .enabled = 0 };

    pub fn probeCount(self: ProbeGrid) u32 {
        return self.counts[0] * self.counts[1] * self.counts[2];
    }
};

// What the main pass writes instead of the shaded surface, for looking at one
// term of it alone. Linear radiance or irradiance over pi in every view, so the
// post pass exposes and tone maps it as it does the frame.
pub const DebugView = enum(u32) {
    shaded = 0,
    // The ambient's diffuse half before the albedo: the probe field where
    // there is one, the hemisphere where there is not. Without occlusion.
    indirect = 1,
    // The light loop alone, every light with its shadow.
    direct = 2,
    // The occlusion multiplying the indirect term, authored and screen-space,
    // as a grey.
    occlusion = 3,
    motion = 4,
};

pub const Debug = extern struct {
    view: DebugView align(16) = .shaded,
    padding: [3]u32 = .{ 0, 0, 0 },
};

// The light block, read by every shaded fragment.
//
// A fixed array rather than a storage buffer sized to the scene. The block is
// read by every fragment, so it wants the constant path; the array's length is
// what the shader's loop is bounded by, and `count` is how much of it holds a
// light. The lanes past `count` are not written, so nothing may read them.
//
// The local shadow matrices are here rather than in a block of their own
// because only the fragment stage reads them, beside the lights that index
// them. The bake does not: it rasterizes through the matrix pushed for each
// layer.
pub const Lights = extern struct {
    // x is how many entries of `lights` are live. The other three lanes are the
    // alignment the array needs either way.
    count: [4]u32 align(16),
    lights: [max_lights]Light,
    local_shadow: LocalShadow,
    // World space to each layer's clip space, from `LocalShadowFit`. Written
    // up to the highest layer a light names; the rest are not read.
    shadow_view_projection: [max_shadow_layers]Mat4,
    ambient: Ambient,
    probe_grid: ProbeGrid,
    debug: Debug,

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

    // Write the frame's local shadow settings and layer matrices. Total for the
    // same reason as `fill`: `frame_set.validate` bounds the count first.
    pub fn fillShadows(self: *Lights, settings: LocalShadow, matrices: []const Mat4) void {
        self.local_shadow = settings;
        @memcpy(self.shadow_view_projection[0..matrices.len], matrices);
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
    view_projection: Mat4 align(16),

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
        .view_projection = mat4(zm.identity()),
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
pub fn vulkanClip(view_projection: anytype) @TypeOf(view_projection) {
    var flipped = view_projection;
    inline for (0..4) |row| flipped[row][1] = -flipped[row][1];
    return flipped;
}

// Moves a framebuffer clip-space projection by `offset` in device coordinates:
// every point's device x and y shift by exactly that, whatever its depth,
// because the shift is added as a multiple of clip w before the divide. Row
// vectors, so column k of the result is column k plus offset[k] times column 3.
pub fn jitterClip(view_projection: anytype, offset: [2]f32) @TypeOf(view_projection) {
    var moved = view_projection;
    inline for (0..4) |row| {
        moved[row][0] += offset[0] * view_projection[row][3];
        moved[row][1] += offset[1] * view_projection[row][3];
    }
    return moved;
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
    flipped.previous_view_projection = vulkanClip(camera.previous_view_projection);
    inline for (0..3) |axis| flipped.ray_up[axis] = -camera.ray_up[axis];
    return .{ .block = flipped };
}
