const std = @import("std");
const materials = @import("binding/materials.zig");

// The host account of what a main-pass fragment stage computes from a material.
// It calls no Vulkan and holds no device state, so the arithmetic the picture
// depends on can be answered by a test rather than by a screenshot.
//
// The formulas are glTF 2.0 Appendix B.3, and a shader is written from the same
// section. Nothing generates one side from the other, so a divergence shows up
// as a golden vector that stops matching rather than as a build error.
//
// It stays in this module although nothing in `src/` calls it, and the reason
// is the record it reads. Every function here interprets a field of
// `materials.MaterialData`, whose packing this module owns and whose
// meaning is fixed by the same specification section. Splitting the two would
// put a layout and its interpretation in different repositories.
//
// A second backend would reuse these bodies unchanged, which usually places a
// file elsewhere. It does not here, for the reason `staging/placement.zig` is
// also neutral arithmetic that stays: what a file serves decides where it
// lives. This serves the packed material record.
//
// What would move it is a second shading model with a host mirror of its own.
// Two mirrors in two repositories is the state this arrangement cannot
// describe.

// The floor the roughness is held above, and a shader holds it above the same
// one. A punctual light is a delta, so at the mirror direction the
// distribution below is 1/(pi * a^2) with a = roughness^2, and the specular
// term approaches intensity / (4 * pi * a^2). The HDR attachment's narrowest
// channel is an unsigned 10-bit float whose largest finite value is
// 2^15 * (1 + 31/32) = 63488 (Khronos Data Format Specification 1.3, section
// "Unsigned 10-bit floating-point numbers"). At this floor a light of unit-ish
// intensity peaks near 6e3, which leaves room for several lights before a
// highlight saturates the format instead of the tone mapper.
pub const min_roughness: f32 = 0.08;

pub const pi: f32 = 3.14159265;

// The dielectric reflectance at normal incidence. glTF 2.0 Appendix B.2.2 fixes
// the index of refraction at 1.5, which is this value, and it is why a non-metal
// carries no reflectance of its own in this material model.
pub const dielectric_f0: f32 = 0.04;

// Samples consumed before lighting. Keeping the two textures named avoids a
// positional pair whose channels can be interchanged without a type error.
pub const MaterialSamples = struct {
    base_colour: [3]f32,
    metallic_roughness: [3]f32,
};

// What a fragment shades with, after the material's factors and its textures
// have been combined. The shader reaches this state before any light is
// considered, so the split is the same one it makes.
pub const Surface = struct {
    base_colour: [3]f32,
    metallic: f32,
    // Already floored. `fromMaterial` is what applies the floor, so a surface
    // built by hand for a test carries whatever it was given.
    roughness: f32,

    // glTF 2.0 section 3.9.2 makes each factor a linear multiplier of its
    // texture and assigns roughness to green and metalness to blue. An absent
    // texture has value 1.0. The presence mask selects that value explicitly;
    // the fallback image's texel is not part of the material arithmetic.
    pub fn fromMaterial(material: materials.MaterialData, samples: MaterialSamples) Surface {
        var base: [3]f32 = undefined;
        inline for (0..3) |channel|
            base[channel] = samples.base_colour[channel] * material.base_colour_factor[channel];

        const has_metallic_roughness =
            material.flags[2] & materials.texture_present.metallic_roughness != 0;
        const metallic_sample = if (has_metallic_roughness) samples.metallic_roughness[2] else 1.0;
        const roughness_sample = if (has_metallic_roughness) samples.metallic_roughness[1] else 1.0;

        return .{
            .base_colour = base,
            .metallic = std.math.clamp(
                metallic_sample * material.metallic_roughness_cutoff[0],
                0,
                1,
            ),
            .roughness = std.math.clamp(
                roughness_sample * material.metallic_roughness_cutoff[1],
                min_roughness,
                1,
            ),
        };
    }

    // The distribution's parameter. Appendix B.2.3: the mapping alpha =
    // roughness^2 is what makes a change in roughness perceptually even.
    pub fn alpha(self: Surface) f32 {
        return self.roughness * self.roughness;
    }

    // Appendix B.3.5. A metal reflects no diffuse light, and its reflectance at
    // normal incidence is its own base colour rather than the dielectric's.
    pub fn diffuseColour(self: Surface) [3]f32 {
        var out: [3]f32 = undefined;
        inline for (0..3) |channel|
            out[channel] = self.base_colour[channel] * (1 - self.metallic);
        return out;
    }

    pub fn f0(self: Surface) [3]f32 {
        var out: [3]f32 = undefined;
        inline for (0..3) |channel|
            out[channel] = std.math.lerp(dielectric_f0, self.base_colour[channel], self.metallic);
        return out;
    }
};

// What the alpha mode decides about a fragment, before anything is lit.
pub const Coverage = struct {
    // The alpha the fragment writes. Only a blended pipeline reads it; a solid
    // one has blending disabled and the value is ignored by the attachment.
    alpha: f32,
    // The fragment contributes nothing and is dropped before shading.
    discarded: bool,
};

// glTF 2.0 section 3.9.4. The alpha is the fourth component of the base colour,
// which is the base-colour factor times the texture's own alpha, and each mode
// reads it differently:
//
// OPAQUE ignores it and the output is fully opaque, so the sample is not even
// consulted. MASK compares it against the cutoff and is fully opaque at or above
// it and fully transparent below, which is why a surviving masked fragment
// writes one rather than the alpha it was tested with. BLEND carries it through
// to the "over" operator.
//
// The mode is the material's first flag lane, packed as the ordinal of
// `MaterialInfo.Rendering.AlphaMode` and pinned to it by the asserts in
// binding/materials.zig. An unknown value cannot arrive: the lane is written from
// that enum and nothing else writes the buffer.
pub fn coverage(material: materials.MaterialData, sampled_alpha: f32) Coverage {
    const alpha = sampled_alpha * material.base_colour_factor[3];
    return switch (material.alphaMode()) {
        .@"opaque" => .{ .alpha = 1, .discarded = false },
        .mask => .{
            .alpha = 1,
            .discarded = alpha < material.metallic_roughness_cutoff[2],
        },
        .blend => .{ .alpha = alpha, .discarded = false },
    };
}

// What the material adds on its own, before any light is considered. The sample
// is display-encoded by the asset but arrives here after sRGB decoding. glTF 2.0
// section 3.9.3 makes its linear RGB a multiplier of the emissive factor; an
// absent texture has value 1.0.
pub fn emissive(material: materials.MaterialData, sampled: [3]f32) [3]f32 {
    const has_texture = material.flags[2] & materials.texture_present.emissive != 0;
    var out: [3]f32 = undefined;
    inline for (0..3) |channel| {
        const texture_value = if (has_texture) sampled[channel] else 1.0;
        out[channel] = texture_value * material.emissive_factor[channel];
    }
    return out;
}

// The Trowbridge-Reitz/GGX distribution, Appendix B.3.2. The Heaviside term is
// the caller's: this is reached only where the light is above the horizon.
pub fn distribution(n_dot_h: f32, alpha: f32) f32 {
    const alpha_squared = alpha * alpha;
    const denominator = n_dot_h * n_dot_h * (alpha_squared - 1) + 1;
    return alpha_squared / (pi * denominator * denominator);
}

// The Smith joint masking-shadowing function in the visibility form of the same
// section: V = G / (4 |N.L| |N.V|), so the microfacet BRDF's own
// 1 / (4 |N.L| |N.V|) is already inside this and is not applied again.
pub fn visibility(n_dot_l: f32, n_dot_v: f32, alpha: f32) f32 {
    const alpha_squared = alpha * alpha;
    const masking = n_dot_l + @sqrt(alpha_squared + (1 - alpha_squared) * n_dot_l * n_dot_l);
    const shadowing = n_dot_v + @sqrt(alpha_squared + (1 - alpha_squared) * n_dot_v * n_dot_v);
    return 1 / (masking * shadowing);
}

// Schlick's approximation, Appendix B.3.4. f90 is 1.0: the grazing reflectance
// of any material approaches white.
pub fn fresnel(f0: [3]f32, v_dot_h: f32) [3]f32 {
    const grazing = 1 - @abs(v_dot_h);
    const weight = grazing * grazing * grazing * grazing * grazing;

    var out: [3]f32 = undefined;
    inline for (0..3) |channel|
        out[channel] = f0[channel] + (1 - f0[channel]) * weight;
    return out;
}

// The directions a fragment shades from, all unit and all in world space.
pub const Geometry = struct {
    normal: [3]f32,
    to_light: [3]f32,
    to_eye: [3]f32,
};

// The world-space basis a tangent-space normal is resolved through. Tangent W
// is the handedness sign carried by glTF's TANGENT attribute.
pub const TangentFrame = struct {
    normal: [3]f32,
    tangent: [4]f32,
};

// glTF 2.0 section 3.9.3 maps RGB from zero-to-one into tangent-space XYZ,
// scales X and Y, then normalizes. Section 3.7.2.1 defines the bitangent as
// cross(normal, tangent) multiplied by tangent W.
pub fn normalFromMaterial(
    material: materials.MaterialData,
    sampled: [3]f32,
    frame: TangentFrame,
) [3]f32 {
    const normal = normalize(frame.normal);
    if (material.flags[2] & materials.texture_present.normal == 0)
        return normal;

    const tangent_direction = [3]f32{ frame.tangent[0], frame.tangent[1], frame.tangent[2] };
    const tangent = normalize(subtract(
        tangent_direction,
        multiply(normal, dot(normal, tangent_direction)),
    ));
    const bitangent = multiply(cross(normal, tangent), frame.tangent[3]);

    const normal_scale = material.metallic_roughness_cutoff[3];
    const tangent_normal = normalize(.{
        (sampled[0] * 2 - 1) * normal_scale,
        (sampled[1] * 2 - 1) * normal_scale,
        sampled[2] * 2 - 1,
    });

    return normalize(add(
        add(multiply(tangent, tangent_normal[0]), multiply(bitangent, tangent_normal[1])),
        multiply(normal, tangent_normal[2]),
    ));
}

// The metallic-roughness BRDF in the form Appendix B.3.5 calls the final BRDF
// for the material. That section builds the mix of a metal and a dielectric
// BRDF, then folds it and states the folded result as what to implement, which
// is what this is. The result is the reflectance, so the caller still multiplies
// by the light's radiance and by N.L.
//
// The fold is not an identity, and the difference is worth knowing before this
// is compared against a renderer written to the unfolded mix. The two specular
// terms agree to 1e-15 over 200 000 random parameter sets, measured, but the
// diffuse terms do not. This one attenuates by `1 - F` with F built from the
// metal-blended f0, where the mix attenuates by `1 - F` from the dielectric f0
// alone and then scales by `1 - metallic`. They meet at metallic 0 and at
// metallic 1 and part in between, by up to 0.075 in reflectance for a bright
// base colour around metallic 0.54.
pub fn brdf(surface: Surface, geometry: Geometry) [3]f32 {
    const half_vector = normalize(add(geometry.to_light, geometry.to_eye));
    const n_dot_l = @max(dot(geometry.normal, geometry.to_light), 0);
    const n_dot_v = @max(dot(geometry.normal, geometry.to_eye), 0);
    const n_dot_h = @max(dot(geometry.normal, half_vector), 0);
    const v_dot_h = @max(dot(geometry.to_eye, half_vector), 0);

    const alpha = surface.alpha();
    const f = fresnel(surface.f0(), v_dot_h);
    const c_diff = surface.diffuseColour();
    const specular = distribution(n_dot_h, alpha) * visibility(n_dot_l, n_dot_v, alpha);

    var out: [3]f32 = undefined;
    inline for (0..3) |channel|
        out[channel] = (1 - f[channel]) * (1.0 / pi) * c_diff[channel] +
            f[channel] * specular;
    return out;
}

// What the environment contributes, resolved to the values one fragment reads
// out of it. Keeping the samples separate from the arithmetic is what lets the
// integration be tested without a cubemap: the shader's three texture reads
// become three fields.
pub const EnvironmentSamples = struct {
    // The cosine-convolved irradiance along the shading normal, from the
    // lambertian cubemap.
    irradiance: [3]f32,
    // The prefiltered GGX radiance along the reflection vector, from the level
    // `specularLod` selects.
    specular: [3]f32,
    // The GGX environment-BRDF term, a scale and a bias for f0, tabulated
    // against (N.V, roughness). Two channels, in that order.
    f_ab: [2]f32,
};

// The split-sum approximation stores one roughness per mip level, spread
// linearly over the chain, so the level is the roughness times the last index.
// This is the mapping the prefilter used, not a choice made here: reading a
// different level than the one a roughness was integrated for returns a
// correctly filtered value of the wrong roughness, which looks like a material
// error rather than a sampling one.
pub fn specularLod(roughness: f32, mip_levels: u32) f32 {
    std.debug.assert(mip_levels > 0);
    return roughness * @as(f32, @floatFromInt(mip_levels - 1));
}

// The reflection of the view direction about the normal, which is where a
// mirror surface sees the environment. `to_eye` points away from the surface,
// so the incident direction is its negation and this is `reflect(-V, N)`.
pub fn reflection(normal: [3]f32, to_eye: [3]f32) [3]f32 {
    const incident = multiply(to_eye, -1);
    return normalize(subtract(incident, multiply(normal, 2 * dot(normal, incident))));
}

// The Fresnel-weighted environment BRDF, including the energy a single-scatter
// microfacet model loses. Fdez-Aguera, "A Multiple-Scattering Microfacet Model
// for Real-Time Image-based Lighting", Journal of Computer Graphics Techniques
// 8(1), 2019; the Khronos reference implementation uses the same two
// expressions in `getIBLGGXFresnel`.
//
// Two things here are deliberately unlike the direct-light Fresnel above.
// The reflectance is raised toward `1 - roughness` before Schlick is applied,
// because a rough surface reflects less at grazing angles than a smooth one and
// plain Schlick does not know that. And the multiple-scattering term adds back
// the energy the single-scatter integral drops, which is what stops a rough
// metal from darkening as roughness grows.
pub fn iblFresnel(f0: [3]f32, roughness: f32, n_dot_v: f32, f_ab: [2]f32) [3]f32 {
    const grazing = 1 - n_dot_v;
    const weight = grazing * grazing * grazing * grazing * grazing;

    var out: [3]f32 = undefined;
    inline for (0..3) |channel| {
        const reflectance = @max(1 - roughness, f0[channel]) - f0[channel];
        const k_s = f0[channel] + reflectance * weight;
        const single = k_s * f_ab[0] + f_ab[1];

        // The fraction of energy the single-scatter term failed to account for,
        // and the average reflectance it is redistributed with. The 21 is the
        // analytic average of the Schlick term over the hemisphere, from the
        // same paper.
        const missing = 1 - (f_ab[0] + f_ab[1]);
        const average = f0[channel] + (1 - f0[channel]) / 21;
        // A perfect reflector's average is exactly one, so a table reading zero
        // at that texel makes this denominator zero with a zero numerator above
        // it. The limit is finite; the arithmetic is not, and the NaN survives
        // an interpolation weight of zero and reaches the picture as black.
        const multiple = missing * single * average /
            @max(1 - average * missing, 1.0e-4);

        out[channel] = single + multiple;
    }
    return out;
}

// The environment's contribution, as glTF 2.0 Appendix B.1 composes a material:
// a linear interpolation of a metal and a dielectric BRDF by metalness. The
// metal reflects the environment tinted by its base colour and has no diffuse
// lobe; the dielectric mixes its irradiance and its reflection by the same
// Fresnel term.
//
// Unlike the direct path, the diffuse term multiplies the base colour itself
// rather than a base colour already scaled by `1 - metallic`: the mixture is
// what removes a metal's diffuse lobe here, so scaling twice would remove it
// twice.
//
// Left as the mixture on purpose, where the direct path uses the form B.3.5
// folds the same mix into. That fold does not carry over. `iblFresnel` is the
// multiple-scattering term and not a plain Schlick, and it is not linear in f0,
// so one blended f0 is not the mix of two. Folding this one moves the picture
// visibly at intermediate metalness, not marginally.
//
// The two paths therefore attenuate diffuse differently there. Each follows the
// construction its Fresnel term was written for, and that is the trade.
pub fn imageBasedLight(surface: Surface, n_dot_v: f32, samples: EnvironmentSamples) [3]f32 {
    const metal_fresnel = iblFresnel(surface.base_colour, surface.roughness, n_dot_v, samples.f_ab);
    const dielectric_fresnel = iblFresnel(
        @splat(dielectric_f0),
        surface.roughness,
        n_dot_v,
        samples.f_ab,
    );

    var out: [3]f32 = undefined;
    inline for (0..3) |channel| {
        const diffuse = samples.irradiance[channel] * surface.base_colour[channel];
        const metal = metal_fresnel[channel] * samples.specular[channel];
        const dielectric = std.math.lerp(
            diffuse,
            samples.specular[channel],
            dielectric_fresnel[channel],
        );
        out[channel] = std.math.lerp(dielectric, metal, surface.metallic);
    }
    return out;
}

// How much of the environment reaches this fragment. glTF 2.0 section 3.9.3
// states that occlusion affects indirect light and that direct lighting is not,
// so this scales `imageBasedLight` alone: not the BRDF a punctual light is
// shaded through, and not the emissive term, which the surface radiates rather
// than receives.
//
// The value is red and the other channels are ignored, per section 5.19.6.
// Section 5.21.3 gives the form below and bounds the strength to [0, 1], so a
// sample in that range keeps the result there and nothing has to clamp.
//
// The absent case is taken from the presence mask rather than from the
// fallback texel. Both answer 1.0, and the mask is what lets the shader skip
// the fetch.
pub fn occlusion(material: materials.MaterialData, sampled_red: f32) f32 {
    if (material.flags[2] & materials.texture_present.occlusion == 0) return 1;
    const strength = material.emissive_factor[3];
    return 1 + strength * (sampled_red - 1);
}

fn dot(a: [3]f32, b: [3]f32) f32 {
    var sum: f32 = 0;
    inline for (0..3) |channel| sum += a[channel] * b[channel];
    return sum;
}

fn add(a: [3]f32, b: [3]f32) [3]f32 {
    var out: [3]f32 = undefined;
    inline for (0..3) |channel| out[channel] = a[channel] + b[channel];
    return out;
}

fn subtract(a: [3]f32, b: [3]f32) [3]f32 {
    var out: [3]f32 = undefined;
    inline for (0..3) |channel| out[channel] = a[channel] - b[channel];
    return out;
}

fn multiply(v: [3]f32, scalar: f32) [3]f32 {
    var out: [3]f32 = undefined;
    inline for (0..3) |channel| out[channel] = v[channel] * scalar;
    return out;
}

fn cross(a: [3]f32, b: [3]f32) [3]f32 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

fn normalize(v: [3]f32) [3]f32 {
    const length = @sqrt(dot(v, v));
    var out: [3]f32 = undefined;
    inline for (0..3) |channel| out[channel] = v[channel] / length;
    return out;
}
