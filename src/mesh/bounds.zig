const std = @import("std");
const vertex = @import("vertex.zig");

const Vec3 = vertex.Vec3;
const Vertex3D = vertex.Vertex3D;

// Object-space bounds of a mesh, in the space its vertices are already in.
//
// For a skinned mesh these describe the bind pose and nothing else. A pose can
// move a vertex anywhere the skeleton reaches, so whoever culls has to treat a
// skinned mesh as unbounded rather than trust these.
pub const Bounds = struct {
    box: Aabb,
    sphere: Sphere,

    // Both are produced together because the sphere is derived from the box.
    // Separate calls would carry a precondition that the box passed in belongs
    // to the same vertices, and no caller can be checked against that; producing
    // them in one place means there is nothing to state.
    pub fn compute(vertices: []const Vertex3D) Bounds {
        const box = Aabb.compute(vertices);
        return .{ .box = box, .sphere = Sphere.aroundBox(box, vertices) };
    }
};

// The cheap rejection test for culling: a world sphere is this centre through
// the model matrix with the radius scaled by the largest axis scale.
pub const Sphere = struct {
    centre: Vec3,
    radius: f32,

    // Centred on the box midpoint with the exact largest distance as the radius.
    // That is tighter than taking the box's corner distance and needs no
    // iteration, though it is not the minimal enclosing sphere.
    //
    // The box must be the box of these vertices, which is why this is only
    // reachable through Bounds.compute.
    fn aroundBox(box: Aabb, vertices: []const Vertex3D) Sphere {
        if (vertices.len == 0) return .{ .centre = @splat(0), .radius = 0 };

        const centre = (box.min + box.max) * @as(Vec3, @splat(0.5));
        var radius_squared: f32 = 0;
        for (vertices) |*item| {
            const offset = item.position - centre;
            radius_squared = @max(radius_squared, @reduce(.Add, offset * offset));
        }
        return .{ .centre = centre, .radius = @sqrt(radius_squared) };
    }
};

// The tight test behind the sphere. Picking transforms the ray into object space
// and slab-tests this after the sphere has rejected the easy misses.
pub const Aabb = struct {
    min: Vec3,
    max: Vec3,

    // An empty mesh has no meaningful extent, and a degenerate box at the origin
    // is what every consumer treats as nothing to draw.
    //
    // A vertex whose position is not a number drops out of the result rather
    // than poisoning it: Zig's @min and @max return the operand that is not NaN,
    // measured on 0.16. The box is therefore that of the finite vertices, which
    // is the useful answer, and a mesh whose positions are all NaN yields a box
    // of NaN because there is no finite one to report.
    pub fn compute(vertices: []const Vertex3D) Aabb {
        if (vertices.len == 0) return .{ .min = @splat(0), .max = @splat(0) };

        var min = vertices[0].position;
        var max = vertices[0].position;
        for (vertices[1..]) |*item| {
            min = @min(min, item.position);
            max = @max(max, item.position);
        }
        return .{ .min = min, .max = max };
    }

    pub fn centre(self: Aabb) Vec3 {
        return (self.min + self.max) * @as(Vec3, @splat(0.5));
    }

    pub fn extent(self: Aabb) Vec3 {
        return self.max - self.min;
    }
};
