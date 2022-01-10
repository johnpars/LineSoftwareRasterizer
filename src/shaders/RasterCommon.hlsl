// Defines
// -----------------------------------------------------

// Hardware specific metrics, based on Nvidia Quadro RTX 8000.
// TEMP: Specs for Radeon Pro 460 GCN
#define NUM_CU              16 // 72
#define NUM_WAVE_PER_CU     40 // 32
#define NUM_LANE_PER_WAVE   64 // 32

#define ZERO_INITIALIZE(type, name) name = (type)0;

// Counter indices
#define ATOMIC_COUNTER_BIN 0

// Structures
// -----------------------------------------------------
struct FragmentData
{
    float4 color;
    float  depth;
    uint   next;
};

struct StrandData
{
    float3 strandPositionOS;
};

struct VertexInput
{
    float vertexID;
    float vertexUV;
};

struct VertexOutput
{
    float4 positionCS;
};

struct SegmentRecord
{
    float2 v0;
    float2 v1;
};

struct SegmentData
{
    // Vertex Indices
    uint vi0;
    uint vi1;
};

struct AABB
{
    float2 min;
    float2 max;

    float2 Center()
    {
        return (min + max) * 0.5;
    }
};

// Helpers
// -----------------------------------------------------
bool WaveIsLastLane()
{
    return WaveGetLaneIndex() == NUM_LANE_PER_WAVE - 1;
}

uint WaveIndex(uint groupIndex)
{
    return groupIndex / NUM_LANE_PER_WAVE;
}

// Signed distance to a line segment.
// Ref: https://www.shadertoy.com/view/3tdSDj
float DistanceToSegmentAndBarycentricCoordinate(float2 P, float2 A, float2 B, out float H)
{
    float2 BA = B - A;
    float2 PA = P - A;

    // Also output the 'barycentric' segment coordinate computed as a bi-product of the coverage.
    H = clamp( dot(PA, BA) / dot(BA, BA), 0.0, 1.0);

    return length(PA - H * BA);
}