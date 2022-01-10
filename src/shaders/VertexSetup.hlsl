#include "RasterCommon.hlsl"

// Inputs
cbuffer Constants : register(b0)
{
    float4x4 _MatrixV;
    float4x4 _MatrixP;
    float4   _VertexParams;
}

StructuredBuffer<VertexInput> _VertexInputBuffer : register(t0);
StructuredBuffer<StrandData>  _StrandDataBuffer  : register(t1);

// Outputs
RWStructuredBuffer<VertexOutput> _VertexOutputBuffer : register(u0);

// Defines
#define _StrandCount           _VertexParams.x
#define _StrandParticleCount   _VertexParams.y
#define _PerStrandVertexCount  _StrandParticleCount
#define _PerStrandSegmentCount (_PerStrandVertexCount - 1)
#define _PerStrandIndexCount   (_PerStrandSegmentCount * 2)
#define _IndexCount            _StrandCount * _PerStrandIndexCount
#define _VertexCount           _StrandCount * _PerStrandVertexCount

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

// Basically a vertex shader.
VertexOutput Vert(VertexInput input)
{
    // Setup the strand iterator.
    uint linearParticleIndex = input.vertexID;
    DECLARE_STRAND(input.vertexID / _StrandParticleCount)

    // Compute the strand index.
    const uint i = strandParticleBegin + (linearParticleIndex % _StrandParticleCount) * strandParticleStride;

    // Read the strand data.
    const StrandData strandData = _StrandDataBuffer[i];

    // Compute the output vertex data.
    VertexOutput output;
    {
        output.positionCS = mul(mul(float4(strandData.strandPositionOS, 1.0), _MatrixV), _MatrixP);
    }
    return output;
}

[numthreads(NUM_LANE_PER_WAVE, 1, 1)]
void VertexSetup(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint i = dispatchThreadID.x;

    if (i >= _VertexCount)
        return;

    // Read the input vertex.
    const VertexInput input = _VertexInputBuffer[i];

    // Invoke the vertex shader and write back to output.
    _VertexOutputBuffer[i] = Vert(input);
}