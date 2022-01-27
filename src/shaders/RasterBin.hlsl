#include "RasterCommon.hlsl"

// Inputs
// ----------------------------------------
cbuffer Constants : register(b0)
{
    float4 _Params0;
    float4 _Params1;
};

ByteAddressBuffer               _SegmentOutputBuffer : register(t0);
StructuredBuffer<SegmentData>   _SegmentDataBuffer   : register(t1);
StructuredBuffer<VertexOutput>  _VertexOutputBuffer  : register(t2);
StructuredBuffer<SegmentRecord> _SegmentRecordBuffer : register(t3);

// Outputs
// ----------------------------------------
RWStructuredBuffer<BinRecord> _BinRecords        : register(u0);
RWBuffer<uint>                _BinRecordsCounter : register(u1);
RWBuffer<uint>                _BinCounters       : register(u2);
RWBuffer<uint>                _BinMinZ           : register(u3);
RWBuffer<uint>                _BinMaxZ           : register(u4);

// Define
// ----------------------------------------
#define _SegmentCount _Params0.x
#define _ScreenParams _Params0.yz
#define _TileSize     _Params0.w
#define _TileSizeSS   2.0 * float2(_TileSize.xx / _ScreenParams)
#define _TileDim      _Params1.xy
#define _CurveSamples _Params1.z

// Utility
// ----------------------------------------
bool ExitThread(uint i)
{
    if (i > _SegmentCount)
        return true;

#if RASTER_CURVE
    // Find the start segment index
    i = 3 * floor(i / 3);
    return !any(_SegmentOutputBuffer.Load3(4 * i));
#else
    // Did the segment pass the clipper?
    return _SegmentOutputBuffer.Load(4 * i) == 0;
#endif
}

// TODO: Do this check in tiled raster space, not NDC space.
bool SegmentsIntersectsBin(uint x, uint y, float2 p0, float2 p1, inout float z)
{
    float2 tileB = float2(x, y);
    float2 tileE = tileB + 1.0;

    // Construct an AABB of this tile.
    AABB aabbTile;
    aabbTile.min = float2(tileB * _TileSizeSS - 1.0);
    aabbTile.max = float2(tileE * _TileSizeSS - 1.0);

    // Get the tile's center.
    float2 center = aabbTile.Center();

    float d = DistanceToSegmentAndTValue(center.xy, p0.xy, p1.xy, z);

    // Compute the segment coverage provided by the segment distance.
    // TODO: Looks like screen params not updated when going from big -> small window size.
    const uint pad = 10;
    float coverage = 1 - step((_TileSize + pad) / _ScreenParams.y, d);

    return any(coverage);
}

// TODO: Do this check in tiled raster space, not NDC space.
bool CurveIntersectsBin(uint x, uint y, float2 controlPoints[4])
{
    float2 tileB = float2(x, y);
    float2 tileE = tileB + 1.0;

    // Construct an AABB of this tile.
    AABB aabbTile;
    aabbTile.min = float2(tileB * _TileSizeSS - 1.0);
    aabbTile.max = float2(tileE * _TileSizeSS - 1.0);

    // Get the tile's center.
    float2 center = aabbTile.Center();

    float unused;
    float d = DistanceToCubicBezierAndTValue(center.xy, controlPoints, unused, _CurveSamples);

    // Compute the segment coverage provided by the segment distance.
    // TODO: Looks like screen params not updated when going from big -> small window size.
    const uint pad = 6;
    float coverage = 1 - step((_TileSize + pad) / _ScreenParams.y, d);

    return any(coverage);
}

void RecordBin(uint binIndex, uint segmentIndex, float t)
{
    // For now, just fire into a record list with global atomics.

    // Update this bin's counter and write back the previous value.
    uint binOffset;
    InterlockedAdd(_BinCounters[binIndex], 1, binOffset);

    // Track the minimum and maximum Z for each bin.
    {
        // TODO: Could maybe just put z into the header
        const SegmentData segment = _SegmentDataBuffer[segmentIndex];

        const VertexOutput v0 = _VertexOutputBuffer[segment.vi0];
        const VertexOutput v1 = _VertexOutputBuffer[segment.vi1];

        const float z0 = v0.positionCS.z * rcp(v0.positionCS.w);
        const float z1 = v1.positionCS.z * rcp(v1.positionCS.w);

        const float2 coords = float2(
            t,
            1 - t
        );

        const float z = INTERP(coords, z0, z1);
        InterlockedMin(_BinMinZ[binIndex], asuint(z));
        InterlockedMax(_BinMaxZ[binIndex], asuint(z));
    }

    // Compute the next valid index in the record buffer.
    uint recordIndex;
    InterlockedAdd(_BinRecordsCounter[0], 1, recordIndex);

    // Write back the record.
    BinRecord record;
    {
        record.segmentIndex = segmentIndex;
        record.binIndex     = binIndex;
        record.binOffset    = binOffset;
    }
    _BinRecords[recordIndex] = record;
}

void GetCurveBoundingBox(float2 controlPoints[4], out uint2 tilesB, out uint2 tilesE)
{
    // Ref: https://www.iquilezles.org/www/articles/bezierbbox/bezierbbox.htm
    const float2 p0 = controlPoints[0];
    const float2 p1 = controlPoints[1];
    const float2 p2 = controlPoints[2];
    const float2 p3 = controlPoints[3];

    float2 mi = min(p0, p3);
    float2 ma = max(p0, p3);

    float2 c = -1 * p0 + 1 * p1;
    float2 b =  1 * p0 - 2 * p1 + 1 * p2;
    float2 a = -1 * p0 + 3 * p1 - 3 * p2 + 1 * p3;

    float2 h = b * b - a * c;

    if (h.x > 0.0)
    {
        h.x = sqrt(h.x);
        float t = (-b.x - h.x) / a.x;

        if (t > 0 && t < 1.0)
        {
            float s = 1.0 - t;
            float q = s*s*s*p0.x + 3.0*s*s*t*p1.x + 3.0*s*t*t*p2.x + t*t*t*p3.x;
            mi.x = min(mi.x, q);
            ma.x = max(ma.x, q);
        }

        t = (-b.x + h.x) / a.x;

        if (t > 0 && t < 1)
        {
            float s = 1.0 - t;
            float q = s*s*s*p0.x + 3.0*s*s*t*p1.x + 3.0*s*t*t*p2.x + t*t*t*p3.x;
            mi.x = min(mi.x,q);
            ma.x = max(ma.x,q);
        }
    }

    if( h.y>0.0 )
    {
        h.y = sqrt(h.y);
        float t = (-b.y - h.y)/a.y;
        if( t>0.0 && t<1.0 )
        {
            float s = 1.0-t;
            float q = s*s*s*p0.y + 3.0*s*s*t*p1.y + 3.0*s*t*t*p2.y + t*t*t*p3.y;
            mi.y = min(mi.y,q);
            ma.y = max(ma.y,q);
        }
        t = (-b.y + h.y)/a.y;
        if( t>0.0 && t<1.0 )
        {
            float s = 1.0-t;
            float q = s*s*s*p0.y + 3.0*s*s*t*p1.y + 3.0*s*t*t*p2.y + t*t*t*p3.y;
            mi.y = min(mi.y,q);
            ma.y = max(ma.y,q);
        }
    }

    AABB aabb;
    aabb.min = mi;
    aabb.max = ma;

    // Transform AABB: NDC -> Tiled Raster Space.
    tilesB = ((aabb.min.xy * 0.5 + 0.5) * _ScreenParams) / _TileSize;
    tilesE = ((aabb.max.xy * 0.5 + 0.5) * _ScreenParams) / _TileSize;

    // Clamp AABB to tiled raster space.
    tilesB = clamp(tilesB, int2(0, 0), _TileDim - 1);
    tilesE = clamp(tilesE, int2(0, 0), _TileDim - 1);
}

void GetSegmentBoundingBox(SegmentRecord segment, out uint2 tilesB, out uint2 tilesE)
{
    // Determine the AABB of the segment.
    AABB aabb;
    aabb.min = min(segment.v0, segment.v1);
    aabb.max = max(segment.v0, segment.v1);

    // Transform AABB: NDC -> Tiled Raster Space.
    tilesB = ((aabb.min.xy * 0.5 + 0.5) * _ScreenParams) / _TileSize;
    tilesE = ((aabb.max.xy * 0.5 + 0.5) * _ScreenParams) / _TileSize;

    // Clamp AABB to tiled raster space.
    tilesB = clamp(tilesB, int2(0, 0), _TileDim - 1);
    tilesE = clamp(tilesE, int2(0, 0), _TileDim - 1);
}

// Kernel (1 Thread Block = 1 SM/CU)
// This kernel is responsible for distributing segment rasterization work.
// It is the first "sync point" of continuous lines into discretized screen space tiles.
// ----------------------------------------
[numthreads(NUM_LANE_PER_WAVE, 1, 1)]
void RasterBin(uint3 dispatchThreadID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    // See note: [NOTE-BINNING-PERSISTENT-THREADS]
#if RASTER_CURVE
    const uint s = dispatchThreadID.x * 3;
#else
    const uint s = dispatchThreadID.x;
#endif

    if (ExitThread(s))
        return;

#if RASTER_CURVE
    float2 controlPoints[4];
    LoadControlPoints(s, _SegmentDataBuffer, _VertexOutputBuffer, controlPoints);

    uint2 tilesB, tilesE;
    GetCurveBoundingBox(controlPoints, tilesB, tilesE);
#else
    // Pick a segment from the ring buffer.
    const SegmentRecord segment = _SegmentRecordBuffer[s];

    uint2 tilesB, tilesE;
    GetSegmentBoundingBox(segment, tilesB, tilesE);
#endif

    // Scalarized fast path for per-bin coverage skip. If bin coverage < 3, skip.
    const bool v_fastPath = ((tilesE.x - tilesB.x) * (tilesE.y - tilesB.y)) <= 2;
    const bool s_fastPath = WaveActiveAllTrue(v_fastPath);

    // Scan the bins within the segment AABB and determine per-bin coverage of the segment.
    for (uint x = tilesB.x; x <= tilesE.x; ++x)
    for (uint y = tilesB.y; y <= tilesE.y; ++y)
    {
        float t = 0;

#if RASTER_CURVE
        if (!s_fastPath && !CurveIntersectsBin(x, y, controlPoints))
#else
        if (!s_fastPath && !SegmentsIntersectsBin(x, y, segment.v0, segment.v1, t))
#endif
           continue;

        // Compute the flatted bin index.
        const uint binIndex = y * _TileDim.x + x;

        RecordBin(binIndex, s, t);
    }
}