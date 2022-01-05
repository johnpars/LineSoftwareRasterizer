// Defines
// -----------------------------------------------------

// Hardware specific metrics, based on Nvidia Quadro RTX 8000.
// TEMP: Specs for Radeon Pro 460 GCN
#define NUM_SM              16 // 72
#define NUM_WARP_PER_SM     40 // 32
#define NUM_THREAD_PER_WARP 64 // 32

#define ZERO_INITIALIZE(type, name) name = (type)0;

// Counter indices
#define ATOMIC_COUNTER_BIN 0

// Structures
// -----------------------------------------------------
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

// Helpers
// -----------------------------------------------------
bool WaveIsLastLane()
{
    return WaveGetLaneIndex() == NUM_THREAD_PER_WARP - 1;
}

uint WaveIndex(uint groupIndex)
{
    return groupIndex / NUM_THREAD_PER_WARP;
}