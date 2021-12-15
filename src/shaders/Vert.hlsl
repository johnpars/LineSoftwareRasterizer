// Bound to the second constant register as the first is occupied by segment setup pass.
cbuffer ConstantsVertex : register(b1)
{
    float4x4 _MatrixV;
    float4x4 _MatrixP;
    float4   _VertexParams;
}

// Bound to the third register as the first two are occupied by the vertex / index buffers.
StructuredBuffer<StrandData>  _StrandDataBuffer : register(t2);

#define _StrandCount           _VertexParams.x
#define _StrandParticleCount   _VertexParams.y
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