#include "RasterCommon.hlsl"
#include "debug/DebugUtility.hlsl"

// Input
cbuffer Constants : register(b0)
{
    float4 _Params0;
};

ByteAddressBuffer               _SegmentOutputBuffer : register(t0);
StructuredBuffer<SegmentRecord> _SegmentRecordBuffer : register(t1);
StructuredBuffer<SegmentData>   _SegmentDataBuffer   : register(t2);
StructuredBuffer<VertexOutput>  _VertexOutputBuffer  : register(t3);

// Output
RWBuffer<uint>                   _CounterBuffer      : register(u0);
RWByteAddressBuffer              _HeadPointerBuffer  : register(u1);
RWStructuredBuffer<FragmentData> _FragmentDataBuffer : register(u2);

// Defines
#define _SegmentCount _Params0.x
#define _ScreenParams _Params0.yz

#define INTERP(coords, a, b) coords.y * a + coords.x * b

uint GetFlattenedPixelIndex(uint x, uint y)
{
    return x + _ScreenParams.x * y;
}

[numthreads(NUM_LANE_PER_WAVE, 1, 1)]
void RasterCoverage(uint3 dti : SV_DispatchThreadID)
{
    const uint i = dti.x;

    if (i >= _SegmentCount)
        return;

    const uint segmentCount = _SegmentOutputBuffer.Load(4 * i);

    if (segmentCount < 1)
        return;

    const SegmentRecord segment = _SegmentRecordBuffer[i];
    const SegmentData   data    = _SegmentDataBuffer[i];

    // Load Vertex Data
    const VertexOutput v0 = _VertexOutputBuffer[data.vi0];
    const VertexOutput v1 = _VertexOutputBuffer[data.vi1];

    // Determine the AABB of the segment.
    AABB aabb;
    aabb.min = min(segment.v0, segment.v1);
    aabb.max = max(segment.v0, segment.v1);

    // Convert to raster space.
    int2 tilesB = (aabb.min.xy * 0.5 + 0.5) * _ScreenParams;
    int2 tilesE = (aabb.max.xy * 0.5 + 0.5) * _ScreenParams;

    // Evaluate the coverage of each pixel contained within the AABB for this segment.
    for (int x = tilesB.x; x <= tilesE.x; ++x)
    {
        for (int y = tilesB.y; y <= tilesE.y; ++y)
        {
            float2 tileB = float2(x, y);
            float2 tileE = tileB + 1.0;

            // Construct an AABB of this pixel.
            AABB aabbTile;
            aabbTile.min = float2(tileB * (2.0 / _ScreenParams) - 1.0);
            aabbTile.max = float2(tileE * (2.0 / _ScreenParams) - 1.0);

            // Get the tile's center.
            float2 center = aabbTile.Center();

            float barycenter;
            const float distance = DistanceToSegmentAndBarycentricCoordinate(center, segment.v0, segment.v1, barycenter);

            // Compute the segment coverage provided by the segment distance.
            float coverage = 1 - smoothstep(0.0, 0.002, distance);

            if (!coverage)
                continue;

            float2 b = float2(
                barycenter,
                1 - barycenter
            );

            const float z = INTERP(b, v0.positionCS.z, v1.positionCS.z);
            const float w = INTERP(b, v0.positionCS.w, v1.positionCS.w);
            const float d = z / w;

            uint fragmentCount;
            InterlockedAdd(_CounterBuffer[0], 1, fragmentCount);

            // Exchange the new head pointer.
            uint next;
            _HeadPointerBuffer.InterlockedExchange(4 * GetFlattenedPixelIndex(x, y), segmentCount, next);

            FragmentData data;
            {
                data.color = float4(ColorCycle(i, _SegmentCount), coverage);
                data.depth = z;
                data.next  = next;
            }
            _FragmentDataBuffer[fragmentCount] = data;
        }
    }
}