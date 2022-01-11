#include "RasterCommon.hlsl"

// Inputs
// ----------------------------------------
cbuffer ConstantsSetup : register(b0)
{
    float4 _Params;
}

ByteAddressBuffer               _SegmentOutputBuffer : register(t0);
StructuredBuffer<SegmentRecord> _SegmentRecordBuffer : register(t1);

// Outputs
// ----------------------------------------
RWBuffer<uint> _CounterBuffer : register(u0); // Counter buffer for inter-block coordination via global atomics.
RWBuffer<uint> _BinCounter    : register(u1);

// Define
// ----------------------------------------
#define _SegmentCount _Params.x
#define _BatchSize    _Params.y
#define _ScreenParams _Params.zw

// Launch 512 threads per processor.
#define WAVE_COUNT 8
#define LANE_COUNT WAVE_COUNT * NUM_LANE_PER_WAVE

// Local
// ----------------------------------------
groupshared uint g_RingBuffer[WAVE_COUNT * NUM_LANE_PER_WAVE]; // Group-sized buffer that contains valid segment indices.
groupshared uint g_RingBufferPrefixScratch[WAVE_COUNT];        // Scratch memory for computation of the group prefix sum.
groupshared uint g_RingBufferCount;                            // Intra-block counter for ring buffer index computation.
groupshared uint g_BatchPos;                                   // Inter-block counter for group coordination of batching.

// Utility
// ----------------------------------------

// Perform a group-wide exclusive prefix sum. Returns the prefix sum of x's register at the current lane.
// TODO: There is likely much better ways to do this, start with this approach for now and optimize later.
uint GroupPrefixSum(uint laneID, uint x)
{
    const uint waveID = WaveIndex(laneID);

    // Just allow all the lanes to assign the wave sum to the LDS, not exactly sure if this is 'better' than having the
    // first lane in the wave assign it, but if we do that then you need to be careful of computing the wave sum beforehand
    // (else-wise all lanes aside from the first one will be inactive and the sum result be invalid). Since it's stored in
    // and SGPR anyway this is probably ok for now.
    // Note: Could also just cache this and subtract it from the scratch at the end, to eliminate the second loop.
    g_RingBufferPrefixScratch[waveID] = WaveActiveSum(x);

    // Allow the first lane in the block to compute the prefix sum on the per-wave value.
    if (laneID == 0)
    {
        uint i;

        // 1) Compute the inclusive prefix sum.
        for (i = 1; i < WAVE_COUNT; ++i)
        {
            g_RingBufferPrefixScratch[i] += g_RingBufferPrefixScratch[i - 1];
        }

        // 2) Shift elements once to the right.
        for (i = WAVE_COUNT; i >= 1; --i)
        {
            g_RingBufferPrefixScratch[i] = g_RingBufferPrefixScratch[i - 1];
        }

        // 3) Replace the first element with the identity to arrive at the exclusive sum.
        g_RingBufferPrefixScratch[0] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // NOTE: Might be faster to compute the prefix sum at the beginning and then write the last lane's result as the wave sum.
    return g_RingBufferPrefixScratch[waveID] + WavePrefixSum(x);
}


// Kernel (1 Thread Block = 1 SM/CU)
// This kernel is responsible for distributing segment rasterization work.
// It is the first "sync point" of continues lines into discretized screen space tiles.
// ----------------------------------------
[numthreads(LANE_COUNT, 1, 1)]
void RasterBin(uint3 dispatchThreadID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    uint batchPos = 0;

    // Persistent thread:
    // Process the segment data until we exhaust the input buffer.
    for (;;)
    {
        // First thread in the block increases the batch counter, and informs the other threads in the block.
        if (groupIndex == 0)
            InterlockedAdd(_CounterBuffer[ATOMIC_COUNTER_BIN], _BatchSize, g_BatchPos);
        GroupMemoryBarrierWithGroupSync();
        batchPos = g_BatchPos;

        // Are we done processing the segments? Exit the thread.
        if (batchPos >= _SegmentCount)
            break;

        // Per-Thread State
        uint bufferCount = 0;
        uint batchEnd    = min(batchPos + _BatchSize, _SegmentCount);

        // Keep processing this batch until it is exhausted.
        do
        {
            // Reset the index for the next round.
            g_RingBuffer[groupIndex] = 0;
            g_RingBufferCount = 0;

            // 1) Input phase. Add valid segments into the ring buffer until it is full.
            // This process will compact or expand the results of the segment setup (clipping / tesselation).
            // and ensure that every thread in the group will have some work to do.
            while(bufferCount < LANE_COUNT && batchPos < batchEnd)
            {
                const uint segmentIndex = batchPos + groupIndex;

                // Begin in a culled state.
                uint segmentOutput = 0;

                // Determine the number of segments emitted by the segment setup stage / clipper at this index.
                // (Currently we assume zero or one, but potentially more in the future if tessellated).
                if (segmentIndex < batchEnd)
                {
                    segmentOutput = _SegmentOutputBuffer.Load(4 * segmentIndex);
                }

                // Compute the exclusive prefix sum of the segment output across the thread block.
                // Use it to determine an index into the ring buffer.
                // TODO: Write a slow guaranteed sequential prefix sum to make sure this isn't broken
                const uint ringBufferIndex = bufferCount + GroupPrefixSum(groupIndex, segmentOutput);

                // Record the segment to the ring buffer.
                if (segmentOutput)
                {
                    g_RingBuffer[ringBufferIndex] = segmentIndex;
                }

                // Update the ring buffer offset with a local atomic.
                InterlockedAdd(g_RingBufferCount, segmentOutput);
                GroupMemoryBarrierWithGroupSync();
                bufferCount = g_RingBufferCount;

                // Advance the batch location.
                batchPos = batchPos + LANE_COUNT;
            }

            // 2) Tiled Raster Phase. Each thread picks up a thread in the ring buffer and determines the bin coverage.
            // NVIDIA proposes each CTA / Block manages its own output queue which is later merged in a coarse raster
            // pass. For now, just fire into a buffer with atomics.

            // Pick a segment from the ring buffer.
            const SegmentRecord segment = _SegmentRecordBuffer[g_RingBuffer[groupIndex]];

            // Determine the AABB of the segment.
            AABB aabb;
            aabb.min = min(segment.v0, segment.v1);
            aabb.max = max(segment.v0, segment.v1);

            // Transform AABB: NDC -> Tiled Raster Space.
            int2 tilesB = ((aabb.min.xy * 0.5 + 0.5) * _ScreenParams) / 16;
            int2 tilesE = ((aabb.max.xy * 0.5 + 0.5) * _ScreenParams) / 16;

            tilesB = clamp(tilesB, int2(0, 0), uint2(80, 45) - 1);
            tilesE = clamp(tilesE, int2(0, 0), uint2(80, 45) - 1);

            // Scan the AABB and determine per-tile coverage of the segment.
            for (uint x = tilesB.x; x <= tilesE.x; ++x)
            for (uint y = tilesB.y; y <= tilesE.y; ++y)
            {
                InterlockedAdd(_BinCounter[y * 80 + x], 1);
            }

            // Pass-thru the buffer.
            bufferCount = 0;
        }
        while(batchPos < batchEnd);

        // --
    }
}