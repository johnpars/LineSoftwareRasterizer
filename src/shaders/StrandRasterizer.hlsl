// Force to interleaved for now
#define LAYOUT_INTERLEAVED 1

Buffer<uint> _VertexBuffer : register(t0); // TODO: UV
Buffer<uint> _IndexBuffer  : register(t1);

struct StrandData
{
    float3 strandPositionOS;
};

StructuredBuffer<StrandData> _StrandDataBuffer : register(t2);

cbuffer Constants : register(b0)
{
    float4x4 _MatrixV;
    float4x4 _MatrixP;
    float4   _ScreenParams;
    float4   _Params0;
}

#define _StrandCount         _Params0.x
#define _StrandParticleCount _Params0.y
#define _TotalSegmentCount   _Params0.z

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

RWTexture2D<float4> _OutputTarget : register(u0);

// Temporary line equation solver to visualize the projection
float Line(float2 P, float2 A, float2 B)
{
    float2 AB = B - A;
    float2 AP = P - A;

    float2 T = normalize(AB);
    float l = length(AB);

    float t = clamp(dot(T, AP), 0.0, l);
    float2 closestPoint = A + t * T;

    float distanceToClosest = 1.0 - (length(closestPoint - P) / 0.005);
    float i = clamp(distanceToClosest, 0.0, 1.0);

    return sqrt(i);
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

[numthreads(8, 8, 1)]
void Main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Convert the dispatch coordinates to the generation space [0, 1]
    const float2 UV = ((float2)dispatchThreadID.xy + 0.5) * _ScreenParams.zw;
    const float2 UVh = -1 + 2 * UV;

    float3 result = _OutputTarget[dispatchThreadID.xy].xyz;

    // Brute force iterate over every segment.
    for (int i = 0; i < _TotalSegmentCount; ++i)
    {
        // Load Indices
        const uint i0 = _IndexBuffer[i + 0];
        const uint i1 = _IndexBuffer[i + 1];

        // Load Vertices
        const uint v0 = _VertexBuffer[i0];
        const uint v1 = _VertexBuffer[i1];

        // Invoke Vertex Shader
        const float4 h0 = Vert(v0);
        const float4 h1 = Vert(v1);

        // Perspective divide
        const float3 p0 = h0.xyz / h0.w;
        const float3 p1 = h1.xyz / h1.w;

        // Accumulate Result
        result = max(result, Line(UVh, p0.xy, p1.xy));
    }

    //for (int i = 0; i < _TotalSegmentCount; ++i)
    //{
    //    const uint v = _VertexBuffer[i];
    //    const float4 h = Vert(v);
    //    const float3 p = h.xyz / h.w;
//
    //    result += length(UVh - p.xy) < 0.005 ? 1.0 : 0.0;
    //}

    _OutputTarget[dispatchThreadID.xy] = float4(result, 1.0);
}
