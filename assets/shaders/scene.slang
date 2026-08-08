// The main pass: one instanced mesh, transformed, textured and lit.
//
// Three descriptor sets, split by how often they are rewritten. Set 0 holds what
// the frame supplies and is bound once with a dynamic offset per ring; set 1
// holds the scene's material array and is bound once; set 2 holds one material's
// textures and is bound when the ordered list reaches a new material.

struct CameraUbo
{
    float4x4 view_projection;
    // xyz is the eye in world space, which the fragment stage needs for the
    // view-dependent half of the BRDF. The fourth lane is not read and exists
    // because the block is aligned to sixteen bytes either way.
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

// Mirrors `material_storage.TexTransform`. Declared but not read: the element
// stride of the array below is what the host writes, and dropping the unused
// tail would step through the material array at 64 bytes instead of 224.
// KHR_texture_transform has no path here yet, so every slot is sampled at the
// mesh's own UV.
struct TexTransform
{
    float4 rs;
    float4 params;
};

// Mirrors `material_storage.MaterialData`. One record per material, indexed by
// what the instance carries rather than by which material set is bound.
struct MaterialData
{
    float4 base_colour_factor;
    // Emissive in xyz, occlusion strength in w.
    float4 emissive_factor;
    // Metallic, roughness, alpha cutoff, normal scale.
    float4 metallic_roughness_cutoff;
    // Alpha mode, double sided, the texture-presence mask, unlit.
    uint4 flags;
    TexTransform tex[5];
};

[[vk::binding(0, 0)]] ConstantBuffer<CameraUbo> camera;
[[vk::binding(1, 0)]] StructuredBuffer<Instance> instances;
[[vk::binding(2, 0)]] ConstantBuffer<LightsUbo> lights;
// One array for the whole frame. Each instance owns a contiguous run in it,
// starting at its own `joint_base`, so several skeletons can be drawn as
// instances of one draw.
[[vk::binding(3, 0)]] StructuredBuffer<float4x4> joints;
[[vk::binding(0, 1)]] StructuredBuffer<MaterialData> materials;
// The environment, prefiltered before the engine saw it. Cosine-convolved
// irradiance along the normal, the GGX integral per roughness step along the
// reflection, and the environment BRDF tabulated against (N.V, roughness).
//
// With no environment loaded all three are bound to black, and every term below
// is linear in the two cubemap samples, so the image-based contribution is
// exactly zero without a branch or a flag saying so.
[[vk::binding(1, 1)]] SamplerCube lambertian_environment;
[[vk::binding(2, 1)]] SamplerCube ggx_environment;
[[vk::binding(3, 1)]] Sampler2D ggx_lut;
[[vk::binding(0, 2)]] Sampler2D base_colour;
[[vk::binding(1, 2)]] Sampler2D metallic_roughness;
[[vk::binding(2, 2)]] Sampler2D normal_texture;
[[vk::binding(3, 2)]] Sampler2D emissive_texture;
[[vk::binding(4, 2)]] Sampler2D occlusion_texture;

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
    float4 tangent : TANGENT;
    float2 uv : TEXCOORD0;
    // Uniform over a primitive, so it is flat rather than interpolated: an
    // interpolated index is a fractional index between two records at every
    // fragment but the vertices. `nointerpolation` lowers to the SPIR-V `Flat`
    // decoration on both the output and the input, which the disassembly shows.
    nointerpolation uint material_index : MATERIAL;
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
Varying place(
    float3 local_position,
    float3 local_normal,
    float4 local_tangent,
    float2 uv,
    Instance instance)
{
    const float4 world = mul(float4(local_position, 1.0), instance.model);

    Varying varying;
    varying.position = mul(world, camera.view_projection);
    varying.world_position = world.xyz;
    // The model's linear part, not its inverse transpose. A normal stays
    // perpendicular only under the latter, and this is where a non-uniform
    // scale would show; the transforms this pass is given are rigid.
    varying.normal = mul(local_normal, (float3x3)instance.model);
    varying.tangent = float4(
        mul(local_tangent.xyz, (float3x3)instance.model),
        local_tangent.w);
    varying.uv = uv;
    varying.material_index = instance.material_index;
    return varying;
}

[shader("vertex")]
Varying vertexMain(
    Vertex vertex,
    uint local_instance : SV_InstanceID,
    uint first_instance : SV_StartInstanceLocation)
{
    const Instance instance = instances[instanceIndex(local_instance, first_instance)];
    return place(vertex.position, vertex.normal.xyz, vertex.tangent, vertex.uv, instance);
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
    const float4 skinned_tangent = float4(
        mul(vertex.tangent.xyz, (float3x3)skin),
        vertex.tangent.w);

    return place(skinned_position, skinned_normal, skinned_tangent, vertex.uv, instance);
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

static const float pi = 3.14159265;

// Mirrors the bits of `material_storage.texture_present` used by this pass.
// Base colour has no bit: a material without one is bound to white.
static const uint texture_present_normal = 1u << 0;
static const uint texture_present_metallic_roughness = 1u << 1;
static const uint texture_present_emissive = 1u << 2;
static const uint texture_present_occlusion = 1u << 3;

// Mirrors the ordinals of `MaterialInfo.Rendering.AlphaMode`, which
// material_storage.zig packs into the first flag lane and pins with asserts.
// OPAQUE is zero and is never compared against, because it is what every branch
// below falls through to.
static const uint alpha_mode_mask = 1u;
static const uint alpha_mode_blend = 2u;

// The floor the roughness is held above. A punctual light is a delta, so at the
// mirror direction the distribution below is 1/(pi * a^2) with a = roughness^2,
// and the whole specular term approaches intensity / (4 * pi * a^2). The HDR
// attachment's narrowest channel is an unsigned 10-bit float, whose largest
// finite value is 2^15 * (1 + 31/32) = 63488 (Khronos Data Format
// Specification 1.3, section "Unsigned 10-bit floating-point numbers"). At this
// floor a light of unit-ish intensity peaks near 6e3, an order of magnitude
// under that, which leaves room for several lights before a highlight saturates
// the format rather than the tone mapper.
static const float min_roughness = 0.08;

// The Trowbridge-Reitz/GGX distribution, glTF 2.0 Appendix B.3.2. The Heaviside
// term is the caller's: this is reached only where the light is above the
// horizon.
float distribution(float n_dot_h, float alpha)
{
    const float alpha_squared = alpha * alpha;
    const float denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
    return alpha_squared / (pi * denominator * denominator);
}

// The Smith joint masking-shadowing function in its visibility form, from the
// same section: V = G / (4 |N.L| |N.V|), so the 1/(4 |N.L| |N.V|) of the
// microfacet BRDF is already inside this and is not applied again.
float visibility(float n_dot_l, float n_dot_v, float alpha)
{
    const float alpha_squared = alpha * alpha;
    const float masking = n_dot_l + sqrt(alpha_squared + (1.0 - alpha_squared) * n_dot_l * n_dot_l);
    const float shadowing = n_dot_v + sqrt(alpha_squared + (1.0 - alpha_squared) * n_dot_v * n_dot_v);
    return 1.0 / (masking * shadowing);
}

// Schlick's approximation, glTF 2.0 Appendix B.3.4. f90 is 1.0: the grazing
// reflectance of any material approaches white.
float3 fresnel(float3 f0, float v_dot_h)
{
    const float grazing = 1.0 - abs(v_dot_h);
    const float weight = grazing * grazing * grazing * grazing * grazing;
    return f0 + (1.0 - f0) * weight;
}

// The metallic-roughness BRDF, glTF 2.0 Appendix B.3.5, in the form that
// section reaches after folding the metal and dielectric mixes together:
//
//   c_diff = base colour * (1 - metallic)
//   f0     = lerp(0.04, base colour, metallic)
//   F      = f0 + (1 - f0) * (1 - |V.H|)^5
//   diffuse  = (1 - F) * (1/pi) * c_diff
//   specular = F * D(a) * V(a)
//
// The dielectric f0 of 0.04 is the fixed index of refraction 1.5 the same
// section states, and it is why a non-metal cannot be given a reflectance here.
float3 brdf(
    float3 base,
    float metallic,
    float alpha,
    float3 normal,
    float3 to_light,
    float3 to_eye)
{
    const float3 half_vector = normalize(to_light + to_eye);
    const float n_dot_l = max(dot(normal, to_light), 0.0);
    const float n_dot_v = max(dot(normal, to_eye), 0.0);
    const float n_dot_h = max(dot(normal, half_vector), 0.0);
    const float v_dot_h = max(dot(to_eye, half_vector), 0.0);

    const float3 f0 = lerp(float3(0.04), base, metallic);
    const float3 f = fresnel(f0, v_dot_h);

    const float3 diffuse = (1.0 - f) * (1.0 / pi) * (base * (1.0 - metallic));
    const float3 specular = f * distribution(n_dot_h, alpha) * visibility(n_dot_l, n_dot_v, alpha);
    return diffuse + specular;
}

// The Fresnel-weighted environment BRDF, with the energy a single-scatter
// microfacet model loses added back. Fdez-Aguera, "A Multiple-Scattering
// Microfacet Model for Real-Time Image-based Lighting", Journal of Computer
// Graphics Techniques 8(1), 2019. Mirrored by `iblFresnel` in shading.zig.
//
// The reflectance is raised toward 1 - roughness before Schlick is applied,
// because a rough surface reflects less at grazing angles than a smooth one and
// plain Schlick does not know that. The second term redistributes the energy the
// single-scatter integral drops, which is what stops a rough metal darkening as
// roughness grows.
float3 iblFresnel(float3 f0, float roughness, float n_dot_v, float2 f_ab)
{
    const float grazing = 1.0 - n_dot_v;
    const float weight = grazing * grazing * grazing * grazing * grazing;

    const float3 reflectance = max(float3(1.0 - roughness), f0) - f0;
    const float3 k_s = f0 + reflectance * weight;
    const float3 single = k_s * f_ab.x + f_ab.y;

    // The 21 is the analytic average of the Schlick term over the hemisphere,
    // from the same paper.
    const float missing = 1.0 - (f_ab.x + f_ab.y);
    const float3 average = f0 + (1.0 - f0) / 21.0;
    // A perfect reflector has an average of exactly one, so a lookup table
    // reading zero at that texel drives this denominator to zero while the
    // numerator is also zero. The quotient has a finite limit there and the
    // hardware produces a NaN, which survives being multiplied by a zero
    // interpolation weight and reaches the attachment as black. Measured on a
    // white base colour with a table whose channels were empty.
    const float3 multiple =
        missing * single * average / max(1.0 - average * missing, 1.0e-4);

    return single + multiple;
}

// The environment's contribution, composed the way glTF 2.0 Appendix B.1
// composes a material: a linear interpolation of a metal and a dielectric BRDF
// by metalness. The metal reflects the environment tinted by its base colour and
// has no diffuse lobe; the dielectric mixes irradiance and reflection by the
// same Fresnel term.
//
// The diffuse term multiplies the base colour itself and not a base colour
// already scaled by 1 - metallic. The mixture is what removes a metal's diffuse
// lobe here, and scaling first would remove it twice.
float3 imageBasedLight(
    float3 base,
    float metallic,
    float roughness,
    float3 normal,
    float3 to_eye)
{
    const float n_dot_v = max(dot(normal, to_eye), 0.0);

    // The prefilter spread one roughness per mip level linearly over the chain,
    // so the level is the roughness times the last index. The count is asked of
    // the descriptor rather than carried as a constant beside it: a constant
    // here and a chain of another depth in the file is a silent mismatch that
    // returns a correctly filtered value of the wrong roughness.
    uint width, height, levels;
    ggx_environment.GetDimensions(0, width, height, levels);
    const float lod = roughness * float(max(levels, 1u) - 1u);

    const float3 reflection = normalize(reflect(-to_eye, normal));
    const float3 specular = ggx_environment.SampleLevel(reflection, lod).rgb;
    const float3 irradiance = lambertian_environment.Sample(normal).rgb;
    const float2 f_ab = ggx_lut.Sample(float2(n_dot_v, roughness)).rg;

    const float3 diffuse = irradiance * base;
    const float3 metal = iblFresnel(base, roughness, n_dot_v, f_ab) * specular;
    const float3 dielectric = lerp(
        diffuse,
        specular,
        iblFresnel(float3(0.04), roughness, n_dot_v, f_ab));

    return lerp(dielectric, metal, metallic);
}

// Direct lighting plus the environment. A surface facing away from every light
// is no longer black: it still receives whatever the environment sends it, which
// is also what makes a metal read as metal, since a metal has no diffuse lobe
// and the reflection is all it has.
//
// Alpha modes are not sampled: this pass has a pipeline for OPAQUE alone.
[shader("fragment")]
float4 fragmentMain(Varying varying) : SV_Target
{
    const MaterialData material = materials[varying.material_index];

    // Section 3.9.2: where both are present the factor is a linear multiplier of
    // the texture. A material with no base-colour image is bound to the neutral
    // white fallback, which is that same rule with the texture at 1.0.
    const float4 sampled = base_colour.Sample(varying.uv);
    const float3 base = sampled.rgb * material.base_colour_factor.rgb;

    // Section 3.9.4: the alpha is the fourth component of the base colour, so
    // the factor multiplies the texture here as it does for the other three
    // channels. MASK is fully opaque at or above the cutoff and fully
    // transparent below it, and the test is made here rather than after the
    // lighting that a dropped fragment would never have needed.
    // Named for the base colour it comes from: `alpha` below is the GGX
    // distribution's parameter and the two are unrelated.
    const uint alpha_mode = material.flags.x;
    const float base_alpha = sampled.a * material.base_colour_factor.a;
    if (alpha_mode == alpha_mode_mask && base_alpha < material.metallic_roughness_cutoff.z)
        discard;

    // Section 3.9.2: roughness is green, metalness is blue, and each multiplies
    // its factor. An absent texture has value 1.0. Select that value from the
    // presence mask rather than assigning material meaning to a fallback texel.
    float metallic_sample = 1.0;
    float roughness_sample = 1.0;
    if ((material.flags.z & texture_present_metallic_roughness) != 0)
    {
        const float4 sampled_mr = metallic_roughness.Sample(varying.uv);
        metallic_sample = sampled_mr.b;
        roughness_sample = sampled_mr.g;
    }

    const float metallic = saturate(metallic_sample * material.metallic_roughness_cutoff.x);
    const float roughness = clamp(
        roughness_sample * material.metallic_roughness_cutoff.y,
        min_roughness,
        1.0);
    const float alpha = roughness * roughness;

    // Interpolating unit directions gives shorter vectors between vertices.
    // glTF 2.0 section 3.7.2.1 defines B = cross(N,T) * tangent.w. Section
    // 3.9.3 maps all three sampled channels to tangent-space XYZ, applies scale
    // to X and Y, then requires normalization before lighting.
    float3 normal = normalize(varying.normal);
    if ((material.flags.z & texture_present_normal) != 0)
    {
        const float3 tangent = normalize(
            varying.tangent.xyz - normal * dot(normal, varying.tangent.xyz));
        const float3 bitangent = cross(normal, tangent) * varying.tangent.w;
        float3 tangent_normal = normal_texture.Sample(varying.uv).rgb * 2.0 - 1.0;
        tangent_normal.xy *= material.metallic_roughness_cutoff.w;
        tangent_normal = normalize(tangent_normal);
        normal = normalize(
            tangent * tangent_normal.x +
            bitangent * tangent_normal.y +
            normal * tangent_normal.z);
    }
    const float3 to_eye = normalize(camera.position.xyz - varying.world_position);

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

        const float lambert = dot(normal, to_light);
        if (lambert <= 0.0) continue;

        const float3 reflected = brdf(base, metallic, alpha, normal, to_light, to_eye);
        radiance += reflected * light.colour * (light.intensity * attenuation * lambert);
    }

    // Section 3.9.3: occlusion attenuates indirect light and leaves direct
    // lighting untouched, so it scales the environment term alone. The loop
    // above and the emissive below are outside it by that rule, and emissive is
    // radiated rather than received in the first place.
    //
    // Section 5.19.6 carries the value in red and requires the other channels to
    // be ignored. Section 5.21.3 gives the strength form, and the strength rides
    // in the emissive vector's fourth lane.
    float occlusion = 1.0;
    if ((material.flags.z & texture_present_occlusion) != 0)
    {
        const float sampled_occlusion = occlusion_texture.Sample(varying.uv).r;
        occlusion = 1.0 + material.emissive_factor.w * (sampled_occlusion - 1.0);
    }
    radiance += occlusion * imageBasedLight(base, metallic, roughness, normal, to_eye);

    // Radiance the surface adds on its own, so it is outside the light loop and
    // unlit by construction. Section 3.9.3 makes the decoded linear sample a
    // multiplier of the factor; an absent texture has value 1.0.
    float3 emissive_sample = float3(1.0);
    if ((material.flags.z & texture_present_emissive) != 0)
        emissive_sample = emissive_texture.Sample(varying.uv).rgb;
    radiance += emissive_sample * material.emissive_factor.rgb;

    // Section 3.9.4: OPAQUE ignores the alpha entirely, and a MASK fragment that
    // reached here passed the cutoff and is fully opaque. Only BLEND carries the
    // value out, and only a blended pipeline enables blending, so what the other
    // two write into this lane is never read.
    return float4(radiance, alpha_mode == alpha_mode_blend ? base_alpha : 1.0);
}
