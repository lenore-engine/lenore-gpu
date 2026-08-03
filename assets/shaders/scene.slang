// The main pass: one instanced mesh, transformed and textured.
//
// Two descriptor sets, split by how often they are rewritten. Set 0 holds what
// the frame supplies and is bound once with a dynamic offset per ring; set 1
// holds what a material supplies and is bound per draw.

struct CameraUbo
{
    float4x4 view_projection;
    // xyz is the eye in world space. The fourth lane is not read here and
    // exists because the block is aligned to sixteen bytes either way.
    float4 position;
};

// Mirrors `uniforms.Light`. Each kind reads only its own fields and the rest
// are zero, so no lane means one thing here and another there.
struct Light
{
    float3 position; // point and spot
    uint kind;       // 0 = directional, 1 = point, 2 = spot
    float3 colour;
    float intensity;
    float3 direction; // unit, the way the light travels; directional and spot
    float range;      // point and spot
    float cos_inner;  // spot
    float cos_outer;  // spot
    float2 padding;
};

// Mirrors `uniforms.max_lights`. The array bounds the loop; `count.x` is how
// much of it holds a light, and the rest is not written by anyone.
static const uint max_lights = 16;

struct LightsUbo
{
    uint4 count; // x = live lights, yzw are the array's alignment
    Light lights[max_lights];
};

// Mirrors `frame_set.Instance`. The padding is declared rather than left to the
// rules, because the stride this struct is stepped through by has to be the one
// the host writes: a disagreement reads the wrong instance from the second one
// on, and the first still looks right.
struct Instance
{
    float4x4 model;
    // Where this instance's joint matrices begin in `joints`. Read by
    // `skinnedVertexMain` and by nothing else.
    uint joint_base;
    // The material and two padding words are scalars and not a `uint3`. A
    // three-component vector aligns to sixteen bytes, which pushes the tail to
    // offset eighty and rounds the element to ninety-six; measured, against the
    // host's eighty.
    uint material_index;
    uint padding_0;
    uint padding_1;
};

[[vk::binding(0, 0)]] ConstantBuffer<CameraUbo> camera;
[[vk::binding(1, 0)]] StructuredBuffer<Instance> instances;
[[vk::binding(2, 0)]] ConstantBuffer<LightsUbo> lights;
// One array for the whole frame. Each instance owns a contiguous run in it,
// starting at its own `joint_base`, so several skeletons can be drawn as
// instances of one draw.
[[vk::binding(3, 0)]] StructuredBuffer<float4x4> joints;
[[vk::binding(0, 1)]] Sampler2D base_colour;

struct Vertex
{
    float3 position : POSITION;
    float2 uv : TEXCOORD0;
    // Delivered as signed normalized floats: the attribute is packed 10-10-10-2
    // and the vertex input stage expands it, so nothing is unpacked here.
    float4 normal : NORMAL;
    float4 tangent : TANGENT;
};

struct Varying
{
    float4 position : SV_Position;
    float3 world_position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};

// `SV_InstanceID` is the index within the draw, not within the buffer: Slang
// declares both `InstanceIndex` and `BaseInstance` and subtracts them, which is
// the D3D semantic. Adding `SV_StartInstanceLocation` back is what makes a
// non-zero `firstInstance` index the ring correctly, and it is why this pass
// requires `shaderDrawParameters`.
uint instanceIndex(uint local_instance, uint first_instance)
{
    return first_instance + local_instance;
}

// The half both entry points share, from a position and a normal already in the
// mesh's local space. Row vectors: the vector is on the left of the matrix,
// which is the convention zmath builds these with and the layout they are
// stored in.
Varying place(float3 local_position, float3 local_normal, float2 uv, float4x4 model)
{
    const float4 world = mul(float4(local_position, 1.0), model);

    Varying varying;
    varying.position = mul(world, camera.view_projection);
    varying.world_position = world.xyz;
    // The model's linear part, not its inverse transpose. A normal stays
    // perpendicular only under the latter, and this is where a non-uniform
    // scale would show; the transforms this pass is given are rigid.
    varying.normal = mul(local_normal, (float3x3)model);
    varying.uv = uv;
    return varying;
}

[shader("vertex")]
Varying vertexMain(
    Vertex vertex,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];
    return place(vertex.position, vertex.normal.xyz, vertex.uv, instance.model);
}

// The skinned entry point. Selected by name at pipeline creation, so the two
// variants are one module and one set of bindings rather than two builds of the
// file.
//
// The skinning stream is binding 1 of the vertex input, locations 4 and 5. The
// weights arrive as floats because the attribute is unorm16 and the vertex input
// stage expands it; nothing is unpacked here.
[shader("vertex")]
Varying skinnedVertexMain(
    Vertex vertex,
    uint4 joint_indices : JOINTS,
    float4 joint_weights : WEIGHTS,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];

    // glTF 2.0 specification, 3.7.3.3: the per-vertex transform is the weighted
    // linear sum of the joint matrices. Summing the matrices and transforming
    // once is not an optimisation of transforming four times and summing; it is
    // what the section states, and the two differ for a normal.
    const uint base = instance.joint_base;
    const float4x4 skin =
        joint_weights.x * joints[base + joint_indices.x] +
        joint_weights.y * joints[base + joint_indices.y] +
        joint_weights.z * joints[base + joint_indices.z] +
        joint_weights.w * joints[base + joint_indices.w];

    // Section 3.7.3.2: only the joint transforms are applied to a skinned mesh,
    // and the transform of the mesh's own node is ignored. The importer leaves
    // these vertices in local space for that reason, so the skin matrix is what
    // places them, and `model` carries the engine's placement of the whole model
    // and nothing from the asset's mesh node.
    const float3 skinned_position = mul(float4(vertex.position, 1.0), skin).xyz;
    // The same matrix on the normal, which is wrong under a non-uniform scale in
    // a joint, for the reason the unskinned path is.
    const float3 skinned_normal = mul(vertex.normal.xyz, (float3x3)skin);

    return place(skinned_position, skinned_normal, vertex.uv, instance.model);
}

// Windowed inverse-square falloff: pure 1/d^2 over most of the radius, reaching
// exactly zero at `range` instead of being cut there.
//
// KHR_lights_punctual, "Range Property", recommends exactly
// `max(min(1 - (d/range)^4, 1), 0) / d^2`, and that is what this is. Frostbite
// squares the window on top of it for a vanishing derivative at the cutoff; the
// extension is the authority for the assets this reads, and the kink it leaves
// is in a term that is already zero there.
//
// Squared distance in, so a caller can cull before taking a root. `d^4` is the
// square of that, so no root is needed here either. The floor on the divisor is
// what a surface at the light's own position lands on.
float rangeAttenuation(float distance_squared, float range)
{
    const float ratio = distance_squared / (range * range);
    return saturate(1.0 - ratio * ratio) / max(distance_squared, 1.0e-4);
}

// The cone, between the two cosines. `cos_inner > cos_outer` is established
// where the cone is built, in `lenore-scene`'s `Light.spot`, which is what keeps
// smoothstep off its undefined equal-edge case.
float spotAttenuation(float3 to_light, Light light)
{
    return smoothstep(light.cos_outer, light.cos_inner, dot(to_light, -light.direction));
}

// Lambert only: no specular, no image-based term. The material this pass binds
// is a base colour and nothing else, so there is no roughness to drive a
// microfacet lobe with and nothing that would read one.
//
// The 1/pi is the Lambertian BRDF's, and it is what makes an intensity in the
// units KHR_lights_punctual states one in produce the radiance that follows from
// them rather than a number scaled by pi.
[shader("fragment")]
float4 fragmentMain(Varying varying) : SV_Target
{
    const float4 albedo = base_colour.Sample(varying.uv);
    // Interpolating two unit normals gives a shorter one between them.
    const float3 normal = normalize(varying.normal);

    float3 radiance = float3(0.0);
    const uint live = min(lights.count.x, max_lights);
    for (uint index = 0; index < live; ++index)
    {
        const Light light = lights.lights[index];

        // Toward the source, which is the direction a dot against the normal
        // wants. The light's own direction is the way it travels.
        float3 to_light;
        float attenuation = 1.0;

        if (light.kind == 0)
        {
            to_light = -light.direction;
        }
        else
        {
            const float3 offset = light.position - varying.world_position;
            const float distance_squared = dot(offset, offset);
            // Out of range before any root or divide.
            if (distance_squared > light.range * light.range) continue;
            to_light = offset * rsqrt(max(distance_squared, 1.0e-8));
            attenuation = rangeAttenuation(distance_squared, light.range);
            if (light.kind == 2) attenuation *= spotAttenuation(to_light, light);
        }

        const float lambert = max(dot(normal, to_light), 0.0);
        radiance += light.colour * (light.intensity * attenuation * lambert);
    }

    return float4(albedo.rgb * radiance * (1.0 / 3.14159265), albedo.a);
}
