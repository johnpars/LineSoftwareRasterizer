#include "RasterCommon.hlsl"

// Work Queue Args:
// Transforms the indirect dispatch block width based on the record count.
// ------------------------------------------------------------------

// Input
Buffer<uint> _BinRecordsCounter0 : register(t0);

// Output
RWBuffer<uint4> _WorkQueueArgs : register(u0);

[numthreads(1, 1, 1)]
void BuildWorkQueueArgs()
{
    const uint buildWorkQueueDispatchSize = (_BinRecordsCounter0[0] + NUM_LANE_PER_WAVE - 1) / NUM_LANE_PER_WAVE;

    _WorkQueueArgs[0] = uint4(
        buildWorkQueueDispatchSize, // Dim X
        1,                          // Dim Y
        1,                          // Dim Z
        0
    );
}

// Work Queue:
// Construct a queue of segment indices, organized with respect to bins.
// ------------------------------------------------------------------

// Input
Buffer<uint>                _BinOffsets         : register(t0);
StructuredBuffer<BinRecord> _BinRecords         : register(t1);
Buffer<uint>                _BinRecordsCounter1 : register(t2);

// Output
RWBuffer<uint> _WorkQueue : register(u0);

// Local
groupshared uint g_RecordCount;

[numthreads(NUM_LANE_PER_WAVE, 1, 1)]
void BuildWorkQueue(uint3 dispatchThreadID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    // Pre-load the total record count into LDS.
    if (groupIndex == 0)
        g_RecordCount = _BinRecordsCounter1[0];
    GroupMemoryBarrierWithGroupSync();

    const uint i = dispatchThreadID.x;

    if (i >= g_RecordCount)
        return;

    // Load the record for this index.
    const BinRecord record = _BinRecords[i];

    // Compute the new index into the work queue.
    uint workQueueIndex = _BinOffsets[record.binIndex] + record.binOffset;

    // Write the bin segment into the work queue at this index.
    _WorkQueue[workQueueIndex] = record.segmentIndex;
}