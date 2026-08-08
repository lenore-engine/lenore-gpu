// The morph prepass: one thread per vertex, resolving a mesh's shape targets
// into the vertex buffer the main pass then draws.
//
// It writes a whole vertex rather than a separate stream of displacements, so
// the graphics pipeline binds this output at binding 0 in place of the mesh's
// own buffer and neither the vertex input nor scene.slang knows that a morph
// happened. A mesh that is both skinned and morphed still skins in the vertex
// stage, over the vertices produced here.

// Mirrors `mesh/vertex.zig`'s GpuVertex, field for field.
//
// Scalars and not a float3: a three-component vector aligns to sixteen bytes,
// which puts the normal at offset sixteen and strides the element by thirty-two
// against the host's twenty-four. Measured against the compiler's reflection,
// which is where tests/reflection.zig holds this.
struct Vertex
{
    float position_x;
    float position_y;
    float position_z;
    // Two f16 texture coordinates. Carried through untouched: the importer reads
    // no morphed texture coordinates and section 3.7.2.2 makes them optional.
    uint uv;
    // 10-10-10-2 snorm, w unused.
    uint normal;
    // 10-10-10-2 snorm, w carries the handedness as plus or minus one.
    uint tangent;
};

// Mirrors `mesh/resource.zig`'s MorphElement: one target's displacement of one
// vertex, interleaved so the target index varies fastest. Scalars for the reason
// above.
struct Delta
{
    float position_x;
    float position_y;
    float position_z;
    float normal_x;
    float normal_y;
    float normal_z;
};

struct Push
{
    uint vertex_count;
    uint target_count;
    // Where this mesh's weights begin in the frame's weight array. One array for
    // the whole frame, as the joint matrices are, so several morphed meshes are
    // several dispatches over one upload.
    uint weight_base;
    // Where this dispatch writes in the destination, which holds one copy of the
    // mesh per frame in flight.
    uint output_base;
};

[[vk::push_constant]] Push push;

[[vk::binding(0, 0)]] StructuredBuffer<Vertex> source;
[[vk::binding(1, 0)]] StructuredBuffer<Delta> deltas;
[[vk::binding(2, 0)]] StructuredBuffer<float> weights;
[[vk::binding(3, 0)]] RWStructuredBuffer<Vertex> morphed;

// One group covers this many vertices, and the host divides the vertex count by
// the same number. Provisional: nothing here has measured 32 or 128 against it,
// and what settles it is a profile of a mesh large enough for the choice to
// show. The value is not repeated on the host, which reads it out of the
// compiler's reflection instead.
static const uint group_size = 64;

// A direction of length zero cannot be normalised, and this is the length below
// which one is treated as degenerate. Mirrors `mesh/vertex.zig`.
static const float degenerate_length = 1e-8;

// Mirrors `normalise` in mesh/vertex.zig, whose fallback for a normal is +Y.
//
// A NaN fails the comparison and takes the fallback. An infinite length is
// refused as well, and that branch is not redundant: rsqrt of it is zero, an
// infinite component times zero is a NaN, and the clamp in packing turns a NaN
// into full scale. Both are reachable, because a displacement is a float an
// asset supplies and nothing has bounded it.
float3 normalizeOrFallback(float3 direction)
{
    const float length_squared = dot(direction, direction);
    if (!(length_squared > degenerate_length * degenerate_length) || isinf(length_squared))
        return float3(0.0, 1.0, 0.0);
    return direction * rsqrt(length_squared);
}

// Vulkan specification, Fixed-Point Data Conversion: a signed normalised value
// of b bits is max(c / (2^(b-1) - 1), -1) for the two's complement integer c.
float unpackSnorm10(uint bits)
{
    // Sign-extended by pushing the ten-bit field to the top of a signed word and
    // shifting arithmetically back, which also drops whatever sat above it.
    const int value = int(bits << 22) >> 22;
    return max(float(value) / 511.0, -1.0);
}

// Vulkan specification, VK_FORMAT_A2B10G10R10_SNORM_PACK32: x occupies bits 0 to
// 9, y 10 to 19, z 20 to 29 and w the top two.
float3 unpackSnorm3x10(uint value)
{
    return float3(
        unpackSnorm10(value),
        unpackSnorm10(value >> 10),
        unpackSnorm10(value >> 20));
}

// The same conversion the other way: round(clamp(v, -1, 1) * 511), stored in
// two's complement.
uint packSnorm10(float value)
{
    const int quantised = int(round(clamp(value, -1.0, 1.0) * 511.0));
    return uint(quantised) & 0x3ffu;
}

// Mirrors `packSnorm3x10_1x2` in mesh/vertex.zig with a zero w, which is what
// the host writes for a normal.
//
// An unmorphed vertex survives the round trip: the value read back was quantised
// by that same host packing, so dividing by 511 and multiplying by it again
// rounds to the integer it started from. A mesh whose weights are all zero
// therefore draws exactly as it would without this pass.
uint packSnorm3x10(float3 direction)
{
    const float3 unit = normalizeOrFallback(direction);
    return packSnorm10(unit.x) |
        (packSnorm10(unit.y) << 10) |
        (packSnorm10(unit.z) << 20);
}

[shader("compute")]
[numthreads(group_size, 1, 1)]
void morphMain(uint3 thread : SV_DispatchThreadID)
{
    const uint vertex = thread.x;
    // The dispatch rounds the vertex count up to whole groups, so the tail of
    // the last one addresses no vertex.
    if (vertex >= push.vertex_count) return;

    Vertex result = source[vertex];
    float3 position = float3(result.position_x, result.position_y, result.position_z);
    float3 normal = unpackSnorm3x10(result.normal);

    // glTF 2.0 specification, 3.7.2.2: a morphed attribute is the base attribute
    // plus the weighted sum of that attribute's displacements.
    const uint first = vertex * push.target_count;
    for (uint target = 0; target < push.target_count; ++target)
    {
        const float weight = weights[push.weight_base + target];
        const Delta delta = deltas[first + target];
        position += weight * float3(delta.position_x, delta.position_y, delta.position_z);
        normal += weight * float3(delta.normal_x, delta.normal_y, delta.normal_z);
    }

    result.position_x = position.x;
    result.position_y = position.y;
    result.position_z = position.z;
    result.normal = packSnorm3x10(normal);
    // The tangent is copied. Section 3.7.2.2 allows a TANGENT displacement and
    // the importer reads none, so there is nothing to apply to it.
    morphed[push.output_base + vertex] = result;
}
