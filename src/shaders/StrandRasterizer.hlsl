#define LAYOUT_INTERLEAVED 1 // Force to interleaved for now
// #define DEBUG_VIEW_VERTICES 1
#define DEBUG_COLOR_STRAND 1

// Maximum representable floating-point number
#define FLT_MAX  3.402823466e+38

struct StrandData
{
    float3 strandPositionOS;
};

struct Vertex
{
    float vertexID;
    float vertexUV;
};

StructuredBuffer<Vertex> _VertexBuffer : register(t0);
Buffer<uint> _IndexBuffer  : register(t1);

StructuredBuffer<StrandData> _StrandDataBuffer : register(t2);

cbuffer Constants : register(b0)
{
    float4x4 _MatrixV;
    float4x4 _MatrixP;
    float4   _ScreenParams;
    float4   _Params0;
}

#define _StrandCount           _Params0.x
#define _StrandParticleCount   _Params0.y
#define _PerStrandVertexCount  _StrandParticleCount
#define _PerStrandSegmentCount (_PerStrandVertexCount - 1)
#define _PerStrandIndexCount   (_PerStrandSegmentCount * 2)
#define _IndexCount            _StrandCount * _PerStrandIndexCount

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

RWTexture2D<float4> _OutputTarget : register(u0);

float3 ColorCycle(uint index, uint count)
{
	float t = frac(index / (float)count);

	// Ref: https://www.shadertoy.com/view/4ttfRn
	float3 c = 3.0 * float3(abs(t - 0.5), t.xx) - float3(1.5, 1.0, 2.0);
	return 1.0 - c * c;
}

// Trivial distance-to-line equation to compute coverage with alpha.
float Line(float2 P, float2 A, float2 B)
{
    float2 AB = B - A;
    float2 AP = P - A;

    float2 T = normalize(AB);
    float l = length(AB);

    float t = clamp(dot(T, AP), 0.0, l);
    float2 closestPoint = A + t * T;

    float distanceToClosest = 1.0 - (length(closestPoint - P) / 0.00075);
    float i = clamp(distanceToClosest, 0.0, 1.0);

    return sqrt(i);
}

float2 ComputeBarycentricCoordinates(float2 P, float2 A, float2 B)
{
    float2 AB = B - A;
    float2 AP = P - A;

    float2 C = A + dot(AP, AB) / dot(AB, AB) * AB;

//    float2 T = normalize(AB);
//    float l = length(AB);
//
//    float t = clamp(dot(T, AP), 0.0, l);
//    float2 closestPoint = A + t * T;

    float t = length(C - A) / length(AB);

    return float2(
            t,
        1 - t
    );
}

// Theoretical Vertex Shader Program
float4 Vert(uint vertexID)
{
    uint linearParticleIndex = vertexID;
    DECLARE_STRAND(vertexID / _StrandParticleCount)

    const uint i = strandParticleBegin + (linearParticleIndex % _StrandParticleCount) * strandParticleStride;

    const StrandData strandData = _StrandDataBuffer[i];

    return mul(mul(float4(strandData.strandPositionOS, 1.0), _MatrixV), _MatrixP);
}

// Theoretical Fragment Shader Program
float3 Frag(uint strandIndex, float2 uv)
{
    const float3 c = DEBUG_COLOR(strandIndex);
    return lerp(c, 1 - c, uv.x);
}

[numthreads(8, 8, 1)]
void Main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Convert the dispatch coordinates to the generation space [0, 1]
    const float2 UV = ((float2)dispatchThreadID.xy + 0.5) * _ScreenParams.zw;
    const float2 UVh = -1 + 2 * UV;

    float3 result = _OutputTarget[dispatchThreadID.xy].xyz;

#if DEBUG_VIEW_VERTICES
    for (int i = 0; i < _StrandCount * _StrandParticleCount; ++i)
    {
        const uint v = _VertexBuffer[i];
        const float4 h = Vert(v);

        if (h.z > 0)
            continue;

        const float3 p = h.xyz / h.w;

        const float l = length(UVh - p.xy);

        result += Frag(i / _StrandParticleCount) * (1.0 - smoothstep(0, 0.005, l));
    }
#else
    // Maintain a Z-Buffer per-thread
    float Z = FLT_MAX;

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
         const float4 h0 = Vert(uint(v0.vertexID));
         const float4 h1 = Vert(uint(v1.vertexID));

         // Clip if behind the projection plane.
         if (h0.z > 0 || h1.z > 0)
            continue;

         // Perspective divide
         const float3 p0 = h0.xyz / h0.w;
         const float3 p1 = h1.xyz / h1.w;

         // Compute the "barycenteric" coordinate on the segment.
         // TODO: Perspective correct
         // (technically redundant computation as we already calculate some of this information in line coverage)
         const float2 coords = ComputeBarycentricCoordinates(UVh, p0.xy, p1.xy);

         // Interpolate Vertex Data
         // TODO: Investigate why I had to flip the coords
         const float2 uv = coords.y * v0.vertexUV + coords.x * v1.vertexUV;

         // Invoke Fragment Shader and mask by Coverage
         result += Frag(i / _PerStrandIndexCount, uv) * Line(UVh, p0.xy, p1.xy);
    }
#endif

    _OutputTarget[dispatchThreadID.xy] = float4(result, 1.0);
}
