// Notes

// [NOTE-BINNING-PERSISTENT-THREADS]
// We explored a persistent thread based work distribution model for the binner but after some initial profiling
// it just does not seem to be worth it for segments. Additionally, a lot of motivation for PS for the binner was to
// help maintain submission order. This is less stringint of a requirement for hair strands since there is an order-independent
// coverage resolve in the fine rasterizer.

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

struct BinRecord
{
    uint segmentIndex;
    uint binIndex;
    uint binOffset;
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

// bool SegmentIntersectsAABB(float3 p0, float3 p1, int x, int y)
// {
//     float2 tileB = float2(x, y);
//     float2 tileE = tileB + 1.0;
//
//     // Construct an AABB of this tile.
//     AABB aabbTile;
//     aabbTile.min = float2(tileB * _TileSizeSS - 1.0);
//     aabbTile.max = float2(tileE * _TileSizeSS - 1.0);
//
//     // Get the tile's center.
//     float2 center = aabbTile.Center();
//
//     // Re-use the coverage computation to factor in strand width.
//     float unused;
//     float d = DistanceToSegmentAndBarycentricCoordinate(center.xy, p0.xy, p1.xy, unused);
//
//     // Compute the segment coverage provided by the segment distance.
//     float coverage = 1 - smoothstep(0.0, 0.02, d);
//
//     return any(coverage);
// }