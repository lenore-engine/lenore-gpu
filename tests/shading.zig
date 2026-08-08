const std = @import("std");
const gpu = @import("lenore-gpu");
const res = @import("lenore-resources");

const testing = std.testing;
const shading = gpu.Shading;

fn expectColour(expected: [3]f32, actual: [3]f32) !void {
    inline for (0..3) |channel|
        try testing.expectApproxEqAbs(expected[channel], actual[channel], 1.0e-6);
}

// Straight at the surface: normal, light and eye all along the same axis, so
// the half vector is the normal and every cosine in the BRDF is one. Every
// expected value below was worked out from glTF 2.0 Appendix B.3 by hand for
// this geometry, where the formulas collapse to arithmetic a reader can follow.
const head_on: shading.Geometry = .{
    .normal = .{ 0, 0, 1 },
    .to_light = .{ 0, 0, 1 },
    .to_eye = .{ 0, 0, 1 },
};

const neutral_samples: shading.MaterialSamples = .{
    .base_colour = @splat(1),
    .metallic_roughness = @splat(1),
};

test "a rough white dielectric reflects its diffuse lobe and a little specular" {
    // alpha is 1, so D is 1/pi and the visibility is 1/4. At normal incidence
    // the Schlick term vanishes and F is the dielectric 0.04 exactly, leaving
    // 0.96/pi of diffuse and 0.04/(4*pi) of specular.
    const surface: shading.Surface = .{
        .base_colour = .{ 1, 1, 1 },
        .metallic = 0,
        .roughness = 1,
    };
    try expectColour(@splat(0.30876059), shading.brdf(surface, head_on));
}

test "a metal has no diffuse lobe and reflects its own colour" {
    // metallic 1 makes c_diff zero and f0 the base colour, so at normal
    // incidence F is the base colour and the whole result is D*V = 1/(4*pi).
    const surface: shading.Surface = .{
        .base_colour = .{ 1, 1, 1 },
        .metallic = 1,
        .roughness = 1,
    };
    try expectColour(@splat(0.07957747), shading.brdf(surface, head_on));
}

test "a black metal reflects nothing at all" {
    // f0 is zero and there is no diffuse component to fall back on, so the
    // surface is black under any light. This is the case that catches a
    // diffuse term left outside the (1 - metallic) weight.
    const surface: shading.Surface = .{
        .base_colour = .{ 0, 0, 0 },
        .metallic = 1,
        .roughness = 1,
    };
    try expectColour(.{ 0, 0, 0 }, shading.brdf(surface, head_on));
}

test "the channels are shaded independently" {
    // A coloured dielectric scales the diffuse lobe per channel while the
    // specular term stays achromatic, because f0 is the same 0.04 everywhere.
    const surface: shading.Surface = .{
        .base_colour = .{ 1, 0.5, 0 },
        .metallic = 0,
        .roughness = 1,
    };
    const shaded = shading.brdf(surface, head_on);
    const specular = 0.04 * 0.07957747;
    try expectColour(.{
        0.96 * 0.31830989 + specular,
        0.96 * 0.31830989 * 0.5 + specular,
        specular,
    }, shaded);
}

test "an oblique light and a smoother surface" {
    // Light at 45 degrees, eye along the normal, roughness 0.5. Nothing
    // collapses here, so this is the vector that would move if any one of D, V
    // or F were replaced by a different approximation.
    const geometry: shading.Geometry = .{
        .normal = .{ 0, 0, 1 },
        .to_light = .{ 0.70710678, 0, 0.70710678 },
        .to_eye = .{ 0, 0, 1 },
    };
    const surface: shading.Surface = .{
        .base_colour = .{ 1, 1, 1 },
        .metallic = 0,
        .roughness = 0.5,
    };
    try expectColour(@splat(0.31251857), shading.brdf(surface, geometry));

    const metal: shading.Surface = .{
        .base_colour = .{ 0.8, 0.8, 0.8 },
        .metallic = 1,
        .roughness = 0.3,
    };
    try expectColour(@splat(0.03094405), shading.brdf(metal, geometry));
}

test "Schlick reaches white at grazing incidence" {
    // The geometry cannot produce a zero V.H, because the half vector bisects
    // the light and the eye, so this is checked on the term itself. f90 is 1.0
    // for every material, which is what makes a grazing surface reflective
    // whatever its base colour.
    try expectColour(.{ 1, 1, 1 }, shading.fresnel(@splat(shading.dielectric_f0), 0));

    // 0.04 + 0.96 * 0.5^5, exactly.
    try expectColour(@splat(0.07), shading.fresnel(@splat(shading.dielectric_f0), 0.5));
}

test "alpha is the square of the roughness" {
    // glTF 2.0 Appendix B.2.3. Squaring at the wrong place is invisible in a
    // single picture and wrong everywhere between the extremes.
    const surface: shading.Surface = .{
        .base_colour = .{ 1, 1, 1 },
        .metallic = 0,
        .roughness = 0.5,
    };
    try testing.expectApproxEqAbs(@as(f32, 0.25), surface.alpha(), 1.0e-6);
}

test "the packed material lands in the lanes the surface reads" {
    // The factors travel as unnamed lanes of two vectors, so a swap between
    // metallic and roughness costs nothing at pack time and changes every
    // shaded pixel. This is the check that stands between them.
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{},
        .factors = .{
            .base_colour = .{ 0.5, 0.25, 0.125, 1 },
            .metallic = 0.75,
            .roughness = 0.375,
        },
        .rendering = .{},
    };
    const packed_data = gpu.MaterialData.fromInfo(&info);

    const surface = shading.Surface.fromMaterial(packed_data, neutral_samples);
    try expectColour(.{ 0.5, 0.25, 0.125 }, surface.base_colour);
    try testing.expectApproxEqAbs(@as(f32, 0.75), surface.metallic, 1.0e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.375), surface.roughness, 1.0e-6);
}

test "the base colour texture multiplies the factor" {
    // glTF 2.0 section 3.9.2: where both are present the factor is a linear
    // multiplier of the sampled value.
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{ .base_colour = .{ .path = "base.png" } },
        .factors = .{ .base_colour = .{ 0.5, 0.5, 0.5, 1 } },
        .rendering = .{},
    };
    const packed_data = gpu.MaterialData.fromInfo(&info);

    const surface = shading.Surface.fromMaterial(packed_data, .{
        .base_colour = .{ 0.4, 0.6, 0.8 },
        .metallic_roughness = @splat(1),
    });
    try expectColour(.{ 0.2, 0.3, 0.4 }, surface.base_colour);
}

test "the metallic-roughness texture multiplies factors from green and blue" {
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{ .metallic_roughness = .{ .path = "surface.png" } },
        .factors = .{ .metallic = 0.8, .roughness = 0.5 },
        .rendering = .{},
    };
    const packed_data = gpu.MaterialData.fromInfo(&info);

    const surface = shading.Surface.fromMaterial(packed_data, .{
        .base_colour = @splat(1),
        .metallic_roughness = .{ 0.9, 0.4, 0.25 },
    });
    try testing.expectApproxEqAbs(@as(f32, 0.2), surface.metallic, 1.0e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.2), surface.roughness, 1.0e-6);

    const another_red = shading.Surface.fromMaterial(packed_data, .{
        .base_colour = @splat(1),
        .metallic_roughness = .{ 0.1, 0.4, 0.25 },
    });
    try testing.expectEqual(surface, another_red);
}

test "an absent metallic-roughness texture leaves both factors unchanged" {
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{},
        .factors = .{ .metallic = 0.75, .roughness = 0.375 },
        .rendering = .{},
    };
    const surface = shading.Surface.fromMaterial(gpu.MaterialData.fromInfo(&info), .{
        .base_colour = @splat(1),
        .metallic_roughness = .{ 0, 0, 0 },
    });

    try testing.expectApproxEqAbs(@as(f32, 0.75), surface.metallic, 1.0e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.375), surface.roughness, 1.0e-6);
}

test "an absent normal texture leaves the geometric normal unchanged" {
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{},
        .factors = .{ .normal_scale = 4 },
        .rendering = .{},
    };
    const normal = shading.normalFromMaterial(
        gpu.MaterialData.fromInfo(&info),
        .{ 1, 0, 0 },
        .{ .normal = .{ 0, 0, 2 }, .tangent = .{ 1, 0, 0, 1 } },
    );
    try expectColour(.{ 0, 0, 1 }, normal);
}

test "the neutral normal texel points along the geometric normal" {
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{ .normal = .{ .path = "normal.png" } },
        .factors = .{},
        .rendering = .{},
    };
    const normal = shading.normalFromMaterial(
        gpu.MaterialData.fromInfo(&info),
        .{ 0.5, 0.5, 1 },
        .{ .normal = .{ 0, 0, 1 }, .tangent = .{ 1, 0, 0, 1 } },
    );
    try expectColour(.{ 0, 0, 1 }, normal);
}

test "normal red and blue resolve through the tangent frame" {
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{ .normal = .{ .path = "normal.png" } },
        .factors = .{ .normal_scale = 0.5 },
        .rendering = .{},
    };
    const normal = shading.normalFromMaterial(
        gpu.MaterialData.fromInfo(&info),
        .{ 0.75, 0.5, 1 },
        .{ .normal = .{ 0, 0, 1 }, .tangent = .{ 1, 0, 0, 1 } },
    );
    try expectColour(.{ 0.24253563, 0, 0.9701425 }, normal);
}

test "tangent handedness flips the normal map's green axis" {
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{ .normal = .{ .path = "normal.png" } },
        .factors = .{},
        .rendering = .{},
    };
    const material = gpu.MaterialData.fromInfo(&info);
    const positive = shading.normalFromMaterial(
        material,
        .{ 0.5, 1, 0.5 },
        .{ .normal = .{ 0, 0, 1 }, .tangent = .{ 1, 0, 0, 1 } },
    );
    const negative = shading.normalFromMaterial(
        material,
        .{ 0.5, 1, 0.5 },
        .{ .normal = .{ 0, 0, 1 }, .tangent = .{ 1, 0, 0, -1 } },
    );
    try expectColour(.{ 0, 1, 0 }, positive);
    try expectColour(.{ 0, -1, 0 }, negative);
}

test "the emissive texture multiplies the factor in linear space" {
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{ .emissive = .{ .path = "lamps.png" } },
        .factors = .{ .emissive = .{ 0.5, 0.25, 0.75 } },
        .rendering = .{},
    };
    try expectColour(
        .{ 0.1, 0.1, 0.6 },
        shading.emissive(gpu.MaterialData.fromInfo(&info), .{ 0.2, 0.4, 0.8 }),
    );
}

test "a material with only an emissive factor emits it" {
    // glTF 2.0 section 3.9.2: an absent texture behaves as 1.0, so nothing is
    // being degraded here and the factor is the whole answer.
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{},
        .factors = .{ .emissive = .{ 0.25, 0.5, 0.75 } },
        .rendering = .{},
    };
    try expectColour(
        .{ 0.25, 0.5, 0.75 },
        shading.emissive(gpu.MaterialData.fromInfo(&info), .{ 0, 0, 0 }),
    );
}

test "another slot's texture does not suppress the emissive factor" {
    // The mask is one word carrying four bits, so reading the wrong one is a
    // silent swap between two materials that both look plausible.
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{ .normal = .{ .path = "normal.png" }, .occlusion = .{ .path = "ao.png" } },
        .factors = .{ .emissive = .{ 1, 0, 0 } },
        .rendering = .{},
    };
    try expectColour(
        .{ 1, 0, 0 },
        shading.emissive(gpu.MaterialData.fromInfo(&info), .{ 0, 0, 0 }),
    );
}

test "the roughness floor is applied where the material is read" {
    // A mirror-smooth material is a delta lobe against a punctual light, and the
    // peak overruns the HDR attachment's narrowest channel rather than the tone
    // mapper. The floor is what a material below it is read as.
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{},
        .factors = .{ .roughness = 0 },
        .rendering = .{},
    };
    const packed_data = gpu.MaterialData.fromInfo(&info);

    const floored = shading.Surface.fromMaterial(packed_data, neutral_samples);
    try testing.expectEqual(shading.min_roughness, floored.roughness);

    // And the floor is not a clamp that also moves a legal value.
    info.factors.roughness = 0.5;
    const untouched = shading.Surface.fromMaterial(gpu.MaterialData.fromInfo(&info), neutral_samples);
    try testing.expectApproxEqAbs(@as(f32, 0.5), untouched.roughness, 1.0e-6);
}

fn surfaceOf(base: [3]f32, metallic: f32, roughness: f32) shading.Surface {
    return .{ .base_colour = base, .metallic = metallic, .roughness = roughness };
}

test "the prefiltered level is the roughness across the chain" {
    // The prefilter spread one roughness per level linearly, so the two ends are
    // the whole agreement: reading level 0 for a rough surface returns a mirror
    // reflection, which looks like the wrong material rather than the wrong mip.
    try testing.expectEqual(@as(f32, 0), shading.specularLod(0, 11));
    try testing.expectEqual(@as(f32, 10), shading.specularLod(1, 11));
    try testing.expectEqual(@as(f32, 5), shading.specularLod(0.5, 11));

    // A single-level chain has nowhere to go, and roughness may not push it off
    // the end.
    try testing.expectEqual(@as(f32, 0), shading.specularLod(1, 1));
}

test "the reflection vector is the mirror direction" {
    const up = [3]f32{ 0, 0, 1 };

    // Looking straight down at a surface reflects straight back.
    try expectColour(up, shading.reflection(up, up));

    // At 45 degrees in the XZ plane the reflection is mirrored in X.
    const oblique = [3]f32{ 0.70710678, 0, 0.70710678 };
    try expectColour(.{ -0.70710678, 0, 0.70710678 }, shading.reflection(up, oblique));

    // Grazing stays unit length, which is what the cubemap lookup needs.
    const grazing = shading.reflection(up, .{ 0.99498744, 0, 0.1 });
    const length = @sqrt(grazing[0] * grazing[0] + grazing[1] * grazing[1] + grazing[2] * grazing[2]);
    try testing.expectApproxEqAbs(@as(f32, 1), length, 1.0e-6);
}

// The property the multiple-scattering term exists for. A perfect mirror
// reflects every photon whatever the tabulated split-sum says, so the two terms
// have to add to exactly one. A single-scatter-only implementation returns
// f_ab.x + f_ab.y here instead, and darkens as roughness grows.
test "a perfect reflector loses no energy at any roughness" {
    const mirror: [3]f32 = @splat(1);
    for ([_]f32{ 0.08, 0.25, 0.5, 0.75, 1.0 }) |roughness| {
        for ([_][2]f32{ .{ 1, 0 }, .{ 0.8, 0.1 }, .{ 0.4, 0.05 }, .{ 0.2, 0.02 } }) |f_ab| {
            const result = shading.iblFresnel(mirror, roughness, 0.7, f_ab);
            inline for (0..3) |channel|
                try testing.expectApproxEqAbs(@as(f32, 1), result[channel], 1.0e-5);
        }
    }
}

test "multiple scattering adds energy the single-scatter term dropped" {
    const gold = [3]f32{ 1.0, 0.766, 0.336 };
    const f_ab = [2]f32{ 0.42, 0.06 };
    const full = shading.iblFresnel(gold, 0.8, 0.6, f_ab);

    inline for (0..3) |channel| {
        // The single-scatter term alone, which is what the expression reduces to
        // without the compensation.
        const reflectance = @max(1 - 0.8, gold[channel]) - gold[channel];
        const grazing = 1 - 0.6;
        const weight = grazing * grazing * grazing * grazing * grazing;
        const single = (gold[channel] + reflectance * weight) * f_ab[0] + f_ab[1];
        try testing.expect(full[channel] > single);
    }
}

test "a metal has no diffuse lobe and a dielectric keeps its irradiance" {
    const base = [3]f32{ 0.9, 0.2, 0.1 };
    const bright: shading.EnvironmentSamples = .{
        .irradiance = .{ 4, 4, 4 },
        .specular = .{ 1, 1, 1 },
        .f_ab = .{ 0.5, 0.05 },
    };
    var dark = bright;
    dark.irradiance = .{ 0, 0, 0 };

    // Fully metallic: the mixture drops the diffuse term entirely, so the
    // irradiance cannot reach the result.
    const metal = surfaceOf(base, 1, 0.4);
    try expectColour(
        shading.imageBasedLight(metal, 0.7, bright),
        shading.imageBasedLight(metal, 0.7, dark),
    );

    // Fully dielectric with a flat environment BRDF: nothing is reflected, so
    // what remains is exactly the irradiance times the base colour.
    const plastic = surfaceOf(base, 0, 0.4);
    const flat: shading.EnvironmentSamples = .{
        .irradiance = .{ 4, 4, 4 },
        .specular = .{ 9, 9, 9 },
        .f_ab = .{ 0, 0 },
    };
    try expectColour(.{ 3.6, 0.8, 0.4 }, shading.imageBasedLight(plastic, 0.7, flat));
}

// The base colour enters the metal branch as its reflectance and the dielectric
// branch as its diffuse albedo. Scaling it by `1 - metallic` before the mixture,
// as the direct path does, removes the metal's reflection colour as well.
test "the metal reflects the environment tinted by its base colour" {
    const base = [3]f32{ 1, 0.5, 0.25 };
    const samples: shading.EnvironmentSamples = .{
        .irradiance = .{ 0, 0, 0 },
        .specular = .{ 1, 1, 1 },
        .f_ab = .{ 1, 0 },
    };

    // f_ab of (1, 0) at normal incidence makes the Fresnel term the base colour
    // itself, so a white environment comes back as the metal's own colour.
    const result = shading.imageBasedLight(surfaceOf(base, 1, 0.0), 1, samples);
    try expectColour(base, result);
}

// A mirror conserves energy whatever the average reflectance is, so the term
// that redistributes the lost energy is invisible there. This pins it on a
// dielectric, where it is a 3 per cent difference: 0.0611 against the 0.0594 an
// implementation using f0 itself as the average returns. The value is the
// paper's expression evaluated in double precision.
test "the redistributed energy uses the hemispherical average reflectance" {
    const result = shading.iblFresnel(@splat(0.04), 0.6, 0.5, .{ 0.35, 0.04 });
    inline for (0..3) |channel|
        try testing.expectApproxEqAbs(@as(f32, 0.061133931), result[channel], 1.0e-6);
}

// Metalness 0 and 1 both hide the difference between attenuating the diffuse
// term once and twice: at 0 the factor is 1 and at 1 the mixture discards the
// branch. Only an intermediate value discriminates, and an intermediate value is
// what a metallic-roughness texture produces at every edge between metal and
// paint.
test "the diffuse term is attenuated by the mixture and not before it" {
    const surface = surfaceOf(.{ 0.8, 0.6, 0.4 }, 0.5, 0.5);
    const samples: shading.EnvironmentSamples = .{
        .irradiance = .{ 2, 2, 2 },
        .specular = .{ 3, 3, 3 },
        .f_ab = .{ 0.4, 0.05 },
    };

    const result = shading.imageBasedLight(surface, 0.6, samples);
    // Scaling the base colour by `1 - metallic` first would give
    // 1.4788, 1.0451, 0.7056 instead.
    try testing.expectApproxEqAbs(@as(f32, 1.850299107), result[0], 1.0e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.323685259), result[1], 1.0e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.891320442), result[2], 1.0e-5);
}

// The case the energy-conservation test above cannot reach: a perfect reflector
// against a lookup table reading zero. The average reflectance is then exactly
// one and the redistribution's denominator is 1 - 1 * 1, with a zero numerator
// over it. This produced a NaN on the device, and a NaN survives an
// interpolation weight of zero, so a fully dielectric surface went black
// wherever its base colour was white. Found in a picture, not in a test.
test "a zero lookup table against a white base colour stays finite" {
    for ([_]f32{ 0.08, 0.5, 1.0 }) |roughness| {
        for ([_]f32{ 0.0, 0.5, 1.0 }) |n_dot_v| {
            const result = shading.iblFresnel(@splat(1), roughness, n_dot_v, .{ 0, 0 });
            inline for (0..3) |channel| {
                try testing.expect(!std.math.isNan(result[channel]));
                try testing.expect(std.math.isFinite(result[channel]));
            }
        }
    }
}

// The same defect one level up, where it actually reached the picture: the metal
// branch is evaluated whatever the metalness is, and only the interpolation
// weight discards it. A non-finite value there is not discarded.
test "a dielectric is unaffected by a metal branch it does not use" {
    const white: [3]f32 = @splat(1);
    const samples: shading.EnvironmentSamples = .{
        .irradiance = .{ 2, 2, 2 },
        .specular = .{ 1, 1, 1 },
        .f_ab = .{ 0, 0 },
    };

    const result = shading.imageBasedLight(surfaceOf(white, 0, 1.0), 0.6, samples);
    // With no environment BRDF the dielectric keeps its irradiance and nothing
    // from the metal branch reaches it.
    try expectColour(.{ 2, 2, 2 }, result);
}

// The scale on everything above, which is why these sit at the end rather than
// with the other texture readers.
fn occludedBy(info: *const res.MaterialInfo, sampled_red: f32) f32 {
    return shading.occlusion(gpu.MaterialData.fromInfo(info), sampled_red);
}

test "at full strength the occlusion is the sampled red channel" {
    // glTF 2.0 section 5.21.3 with the default strength of 1: the expression
    // 1 + 1 * (r - 1) is r, so the texture passes through untouched.
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{ .occlusion = .{ .path = "ao.png" } },
        .factors = .{},
        .rendering = .{},
    };
    try testing.expectApproxEqAbs(@as(f32, 0.25), occludedBy(&info, 0.25), 1.0e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), occludedBy(&info, 0), 1.0e-6);
}

test "the strength interpolates between no occlusion and the sample" {
    // Half strength over a fully occluded texel: 1 + 0.5 * (0 - 1) = 0.5, and
    // over a half-occluded one 1 + 0.5 * (0.5 - 1) = 0.75.
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{ .occlusion = .{ .path = "ao.png" } },
        .factors = .{ .occlusion_strength = 0.5 },
        .rendering = .{},
    };
    try testing.expectApproxEqAbs(@as(f32, 0.5), occludedBy(&info, 0), 1.0e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.75), occludedBy(&info, 0.5), 1.0e-6);
}

test "a strength of zero leaves the environment unscaled" {
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{ .occlusion = .{ .path = "ao.png" } },
        .factors = .{ .occlusion_strength = 0 },
        .rendering = .{},
    };
    for ([_]f32{ 0, 0.5, 1 }) |sample|
        try testing.expectApproxEqAbs(@as(f32, 1), occludedBy(&info, sample), 1.0e-6);
}

test "a material with a strength and no occlusion texture does not occlude" {
    // The presence mask decides, not the texel. A slot bound to its white
    // fallback would answer the same, so this is what separates reading the mask
    // from trusting whatever image the slot happens to hold.
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{},
        .factors = .{ .occlusion_strength = 1 },
        .rendering = .{},
    };
    try testing.expectApproxEqAbs(@as(f32, 1), occludedBy(&info, 0), 1.0e-6);
}

test "another slot's texture does not switch occlusion on" {
    // Occlusion is the fourth bit of a word carrying four, and reading the wrong
    // one darkens every material that has a normal or an emissive map.
    var info: res.MaterialInfo = .{
        .name = "",
        .textures = .{
            .normal = .{ .path = "normal.png" },
            .emissive = .{ .path = "lamps.png" },
            .metallic_roughness = .{ .path = "mr.png" },
        },
        .factors = .{ .occlusion_strength = 1 },
        .rendering = .{},
    };
    try testing.expectApproxEqAbs(@as(f32, 1), occludedBy(&info, 0), 1.0e-6);
}
