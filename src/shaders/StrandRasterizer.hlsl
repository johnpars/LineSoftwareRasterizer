#include "Utility.hlsl"

// Structures
// -----------------------------------------------------
struct StrandData
{
    float3 strandPositionOS;
};

struct Vertex
{
    float vertexID;
    float vertexUV;
};

struct Fragment
{
    float3 a;
    float  t;
    float  z;
};

struct VertexOutput
{
    float2 texcoord;
    float3 positionOS;
    float4 positionCS;
};

struct FragmentInput
{
};

// Buffers
// -----------------------------------------------------
StructuredBuffer<Vertex>     _VertexBuffer     : register(t0);
Buffer<uint>                 _IndexBuffer      : register(t1);
StructuredBuffer<StrandData> _StrandDataBuffer : register(t2);
RWTexture2D<float4>          _OutputTarget     : register(u0);

cbuffer Constants : register(b0)
{
    float4x4 _MatrixV;
    float4x4 _MatrixP;
    float4   _ScreenParams;
    float4   _Params0;
}

// Defines
// -----------------------------------------------------
#define INTERP(coords, a, b) coords.y * a + coords.x * b
#define ZERO_INITIALIZE(type, name) name = (type)0;

// #define LAYOUT_INTERLEAVED 1 // Force to interleaved for now - however for OBJ needs to be sequential
#define DEBUG_COLOR_STRAND 1
//#define OPAQUE 1
#define PERSPECTIVE_CORRECT_INTERPOLATION 1

// Maximum representable floating-point number
#define FLT_MAX  3.402823466e+38

#define _StrandCount           _Params0.x
#define _StrandParticleCount   _Params0.y
#define _PerStrandVertexCount  _StrandParticleCount
#define _PerStrandSegmentCount (_PerStrandVertexCount - 1)
#define _PerStrandIndexCount   (_PerStrandSegmentCount * 2)
#define _IndexCount            _StrandCount * _PerStrandIndexCount

// The maximum number of fragments to correctly blend per-pixel.
#define LAYERS_PER_PIXEL 32

#if LAYOUT_INTERLEAVED
    #define DECLARE_STRAND(x)							\
        const uint strandIndex = x;						\
        const uint strandParticleBegin = strandIndex;	\
        const uint strandParticleStride = _StrandCount;	\
        const uint strandParticleEnd = strandParticleBegin + strandParticleStride * _StrandParticleCount;
#else
    #define DECLARE_STRAND(x)													\
        const uint strandIndex = x;												\
        const uint strandParticleBegin = strandIndex * _StrandParticleCount;	\
        const uint strandParticleStride = 1;									\
        const uint strandParticleEnd = strandParticleBegin + strandParticleStride * _StrandParticleCount;
#endif

#if DEBUG_COLOR_STRAND
    #define DEBUG_COLOR(x) ColorCycle(x, _StrandCount)
#else
    #define DEBUG_COLOR(x) ColorCycle(x, _StrandCount * _StrandParticleCount)
#endif

// Signed distance to a line segment.
// Ref: https://www.shadertoy.com/view/3tdSDj
float ComputeSegmentCoverageAndBarycentericCoordinate(float2 P, float2 A, float2 B, out float H)
{
    float2 BA = B - A;
    float2 PA = P - A;

    // Also output the 'barycentric' segment coordinate computed as a bi-product of the coverage.
    H = clamp( dot(PA, BA) / dot(BA, BA), 0.0, 1.0);

    return length(PA - H * BA);
}

void InitializeBlendingArray(inout Fragment B[LAYERS_PER_PIXEL + 1])
{
    // Create the default fragment.
    // (These defaults are important due to how we merge for memory compression).
    Fragment F;
    F.a = 0;
    F.t = 1;
    F.z = -FLT_MAX;

    for (int i = 0; i < LAYERS_PER_PIXEL + 1; i++)
    {
        B[i] = F;
    }
}

// Implementation of "Multi-Layer Alpha Blending"
// Ref: https://www.intel.com/content/dam/develop/external/us/en/documents/i3d14-mlab-preprint.pdf
void InsertFragment(in Fragment F, inout Fragment B[LAYERS_PER_PIXEL + 1])
{
    // 1-Pass bubble sort to insert the fragment.
    Fragment temp, merge;
    for (int i = 0; i < LAYERS_PER_PIXEL + 1; i++)
    {
        if (F.z >= B[i].z)
        {
            temp = B[i];
            B[i] = F;
            F    = temp;
        }
    }

    // Compression (merge the last two rows since we have a fixed memory size).
    const int m = LAYERS_PER_PIXEL;
    merge.a = B[m - 1].a + B[m].a * B[m - 1].t;
    merge.t = B[m - 1].t * B[m].t;
    merge.z = B[m - 1].z;
    B[m - 1] = merge;
}

// Theoretical Vertex Shader Program
float4 Vert(uint vertexID, out float3 positionOS)
{
    uint linearParticleIndex = vertexID;
    DECLARE_STRAND(vertexID / _StrandParticleCount)

    const uint i = strandParticleBegin + (linearParticleIndex % _StrandParticleCount) * strandParticleStride;

    const StrandData strandData = _StrandDataBuffer[i];

    positionOS = strandData.strandPositionOS;

    return mul(mul(float4(strandData.strandPositionOS, 1.0), _MatrixV), _MatrixP);
}

float rand(float co) { return frac(sin(co*(91.3458)) * 47453.5453); }

// Theoretical Fragment Shader Program
float4 Frag(uint strandIndex, float2 uv, float3 positionOS)
{
    const float3 c = DEBUG_COLOR(strandIndex);
    return float4(lerp(c, 1 - c, uv.x), 0.4); // float4(4 * pow(0.5 * positionOS + 0.5, 3), 0.3); // lerp(c, 1 - c, uv.x);
}

[numthreads(8, 8, 1)]
void BruteForce(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Convert the dispatch coordinates to the generation space [0, 1]
    const float2 UV = ((float2)dispatchThreadID.xy + 0.5) * _ScreenParams.zw;
    const float2 UVh = -1 + 2 * UV;

    float3 result = _OutputTarget[dispatchThreadID.xy].xyz;

#if OPAQUE
    // Maintain a Z-Buffer per-thread
    float Z = -FLT_MAX;
#else
    // Create and initialize the blending array.
    Fragment B[LAYERS_PER_PIXEL + 1];
    InitializeBlendingArray(B);
#endif

    // Brute force iterate over every segment.
    // Emulates the rasterization of line strips.
    for (uint i = 0; i < _IndexCount; i += 2)
    {
         // Load Indices
         const uint i0 = _IndexBuffer[i + 0];
         const uint i1 = _IndexBuffer[i + 1];

         // Load Vertices
         const Vertex v0 = _VertexBuffer[i0];
         const Vertex v1 = _VertexBuffer[i1];

         // Invoke Vertex Shader
         float3 positionOS0, positionOS1;
         const float4 h0 = Vert(uint(v0.vertexID), positionOS0);
         const float4 h1 = Vert(uint(v1.vertexID), positionOS1);

         // Clip if behind the projection plane.
         if (h0.z > 0 || h1.z > 0)
            continue;

         // Perspective divide
         float3 p0 = h0.xyz / h0.w;
         float3 p1 = h1.xyz / h1.w;

         // Compute the "barycenteric" coordinate on the segment.
         float coord;
         float d = ComputeSegmentCoverageAndBarycentericCoordinate(UVh, p0.xy, p1.xy, coord);

         // Compute the segment coverage provided by the segment distance.
         float coverage = 1 - smoothstep(0.0, 0.001, d);

         float2 coords = float2(
            coord,
            1 - coord
         );

#if PERSPECTIVE_CORRECT_INTERPOLATION
         // Ensure perspective correct coordinates.
         const float2 w = rcp(float2(h0.w, h1.w));
         coords = (coords * w) / dot(coords, w);
#endif

         // Interpolate Vertex Data
         const float2 uv         = INTERP(coords, v0.vertexUV, v1.vertexUV);
         const float3 positionOS = INTERP(coords, positionOS0, positionOS1);
         const float  z          = INTERP(coords, p0.z,        p1.z);

         // Invoke Fragment Shader and mask by Coverage
         // TODO: Clean up the mess
#if OPAQUE
         if (coverage > 0 && z > Z)
#else
         if (any(coverage))
#endif
         {
#if OPAQUE
            result = Frag(i / _PerStrandIndexCount, uv, positionOS).rgb * coverage;
            Z = z;
#else
            float4 fragmentValue = Frag(i / _PerStrandIndexCount, uv, positionOS);

            float alpha = coverage * fragmentValue.a;

            Fragment f;
            f.a = fragmentValue.rgb * alpha;
            f.t = 1.0 - alpha;
            f.z = z;

            InsertFragment(f, B);
#endif
         }
    }

    // Composite the blending array.
#if !OPAQUE
    result = 0;

    float transmittance = 1;

    for (int k = 0; k < LAYERS_PER_PIXEL + 1; k++)
    {
        result = result + transmittance * B[k].a;
        transmittance *= B[k].t;
    }
#endif

    _OutputTarget[dispatchThreadID.xy] = float4(result, 1.0);
}

[numthreads(16, 1, 1)]
void CoarsePass(uint3 dispatchThreadID : SV_DispatchThreadID)
{
}