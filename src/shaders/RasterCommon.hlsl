// Notes

// [NOTE-BINNING-PERSISTENT-THREADS]
// We explored a persistent thread based work distribution model for the binner but after some initial profiling
// it just does not seem to be worth it for segments. Additionally, a lot of motivation for PS for the binner was to
// help maintain submission order. This is less stringint of a requirement for hair strands since there is an order-independent
// coverage resolve in the fine rasterizer.

// Defines
// -----------------------------------------------------

// Hardware specific metrics, based on Nvidia Quadro RTX 8000.
#define NUM_CU              72
#define NUM_WAVE_PER_CU     32
#define NUM_LANE_PER_WAVE   32

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
    float  texCoord;
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
float DistanceToSegmentAndTValueSq(float2 P, float2 A, float2 B, out float T)
{
    float2 BA = B - A;
    float2 PA = P - A;

    // Also output the 'barycentric' segment coordinate computed as a bi-product of the coverage.
    T = clamp( dot(PA, BA) / dot(BA, BA), 0.0, 1.0);

    const float2 V = PA - T * BA;
    return dot(V, V);
}

float DistanceToSegmentAndTValue(float2 P, float2 A, float2 B, out float T)
{
    return sqrt(DistanceToSegmentAndTValueSq(P, A, B, T));
}

float DistanceToCubicBezierAndTValue(float2 P, float2 controlPoints[4], out float T, uint sampleCount = 20)
{
    const float2 A = controlPoints[0];
    const float2 B = controlPoints[1];
    const float2 C = controlPoints[2];
    const float2 D = controlPoints[3];

    float2 a = A;

    float2 res = float2(1e10, 0.0);

    for (uint i = 1; i < sampleCount; ++i)
    {
        float t = (float)i / ((float)sampleCount - 1.0);
        float s = 1 - t;

        // Evaluate the cubic.
        float2 b = (A * s * s * s) + (B * 3 * s * s * t) + (C * 3 * s * t * t) + (D * t * t * t);

        // Sample the distance to this segment.
        float unused;
        float d = DistanceToSegmentAndTValueSq(P, a, b, unused);

        if (d < res.x)
            res = float2(d, t);

        a = b;
    }

    T = res.y;

    return sqrt(res.x);
}