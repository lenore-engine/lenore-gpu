// The post pass geometry: one triangle covering the screen, generated from the
// vertex index with no buffer bound.
//
// Three vertices rather than a quad's four. A quad's diagonal splits the screen
// into two triangles, and every pixel along it is rasterized by both, so the
// fragment stage runs twice on that seam. One oversized triangle has no seam.

struct Varying
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
Varying vertexMain(uint index : SV_VertexID)
{
    // 0 -> (0,0), 1 -> (2,0), 2 -> (0,2) in UV, which is (-1,-1), (3,-1),
    // (-1,3) in clip space: the triangle reaches one screen width past the
    // right edge and one height past the bottom, so the visible square is
    // covered exactly once.
    const float2 uv = float2((index << 1) & 2, index & 2);

    Varying varying;
    varying.uv = uv;
    varying.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return varying;
}

// The HDR scene target, written by the main pass and read here.
[[vk::binding(0, 0)]]
Sampler2D scene;

[shader("fragment")]
float4 fragmentMain(Varying varying) : SV_Target
{
    return float4(scene.Sample(varying.uv).rgb, 1.0);
}
