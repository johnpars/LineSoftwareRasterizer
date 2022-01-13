#include "RasterCommon.hlsl"

// Inputs
// ----------------------------------------
cbuffer Constants : register(b0)
{
    float4 _Params0;
    float4 _Params1;
};

ByteAddressBuffer               _SegmentOutputBuffer : register(t0);
StructuredBuffer<SegmentRecord> _SegmentRecordBuffer : register(t1);

// Outputs
// ----------------------------------------
RWStructuredBuffer<BinRecord> _BinRecords        : register(u0);
RWBuffer<uint>                _BinRecordsCounter : register(u1);
RWBuffer<uint>                _BinCounters       : register(u2);

// Define
// ----------------------------------------
#define _SegmentCount _Params0.x
#define _ScreenParams _Params0.yz
#define _TileSize     _Params0.w
#define _TileSizeSS   2.0 * float2(_TileSize.xx / _ScreenParams)
#define _TileDim      _Params1.xy

// Local
// ----------------------------------------

// Utility
// ----------------------------------------
bool ExitThread(uint i)
{
    if (i > _SegmentCount)
        return true;

    // Did the segment pass the clipper?
    return _SegmentOutputBuffer.Load(4 * i) == 0;
}

// TODO: Do this check in tiled raster space, not NDC space.
bool SegmentsIntersectsBin(uint x, uint y, float2 p0, float2 p1)
{
    float2 tileB = float2(x, y);
    float2 tileE = tileB + 1.0;

    // Construct an AABB of this tile.
    AABB aabbTile;
    aabbTile.min = float2(tileB * _TileSizeSS - 1.0);
    aabbTile.max = float2(tileE * _TileSizeSS - 1.0);

    // Get the tile's center.
    float2 center = aabbTile.Center();

    // Re-use the coverage computation to factor in strand width.
    float unused;
    float d = DistanceToSegmentAndTValue(center.xy, p0.xy, p1.xy, unused);

    // Compute the segment coverage provided by the segment distance.
    // TODO: Looks like screen params not updated when going from big -> small window size.
    const uint pad = 2;
    float coverage = 1 - step((_TileSize + pad) / _ScreenParams.y, d);

    return any(coverage);
}

void RecordBin(uint binIndex, uint segmentIndex)
{
    // For now, just fire into a record list with global atomics.

    // Update this bin's counter and write back the previous value.
    uint binOffset;
    InterlockedAdd(_BinCounters[binIndex], 1, binOffset);

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

// Kernel (1 Thread Block = 1 SM/CU)
// This kernel is responsible for distributing segment rasterization work.
// It is the first "sync point" of continuous lines into discretized screen space tiles.
// ----------------------------------------
[numthreads(NUM_LANE_PER_WAVE, 1, 1)]
void RasterBin(uint3 dispatchThreadID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    // See note: [NOTE-BINNING-PERSISTENT-THREADS]
    const uint s = dispatchThreadID.x;

    if (ExitThread(s))
        return;

    // Pick a segment from the ring buffer.
    const SegmentRecord segment = _SegmentRecordBuffer[s];

    // TODO: This might be interesting: https://www.iquilezles.org/www/articles/bezierbbox/bezierbbox.htm
    // Could potentially reduce the search domain and reduce the count of per-bin bezier coverage evaluations.

    // Determine the AABB of the segment.
    AABB aabb;
    aabb.min = min(segment.v0, segment.v1);
    aabb.max = max(segment.v0, segment.v1);

    // Transform AABB: NDC -> Tiled Raster Space.
    int2 tilesB = ((aabb.min.xy * 0.5 + 0.5) * _ScreenParams) / _TileSize;
    int2 tilesE = ((aabb.max.xy * 0.5 + 0.5) * _ScreenParams) / _TileSize;

    // Clamp AABB to tiled raster space.
    tilesB = clamp(tilesB, int2(0, 0), _TileDim - 1);
    tilesE = clamp(tilesE, int2(0, 0), _TileDim - 1);

    // Scalarized fast path for per-bin coverage skip. If bin coverage < 3, skip.
    const bool v_fastPath = ((tilesE.x - tilesB.x) * (tilesE.y - tilesB.y)) <= 2;
    const bool s_fastPath = WaveActiveAllTrue(v_fastPath);

    // Scan the bins within the segment AABB and determine per-bin coverage of the segment.
    for (uint x = tilesB.x; x <= tilesE.x; ++x)
    for (uint y = tilesB.y; y <= tilesE.y; ++y)
    {
        // TODO: On-chip tesselation.

        if (!s_fastPath && !SegmentsIntersectsBin(x, y, segment.v0, segment.v1))
           continue;

        // Compute the flatted bin index.
        const uint binIndex = y * _TileDim.x + x;

        RecordBin(binIndex, s);
    }
}