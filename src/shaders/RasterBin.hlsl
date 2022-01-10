#include "RasterCommon.hlsl"

// Inputs
// ----------------------------------------
cbuffer ConstantsSetup : register(b0)
{
    float4 _Params;
}

// TODO: If we end up not doing tesselation in the segment setup, we can go from uint -> bool. We should also switch
// to ByteAddressBuffer regardless.
StructuredBuffer<uint> _SegmentOutputBuffer : register(t0);

// Outputs
// ----------------------------------------
RWBuffer<uint> _CounterBuffer : register(u0); // Counter buffer for inter-block coordination via global atomics.
RWBuffer<uint> _RingBuffer    : register(u1); // Temporary buffer to view memory dump of a block's LDS ring buffer

// Define
// ----------------------------------------
#define _SegmentCount _Params.x
#define _BatchSize    _Params.y

// Launch 512 threads per processor.
#define WAVE_COUNT 8

// Local
// ----------------------------------------
groupshared uint g_RingBuffer[WAVE_COUNT * NUM_LANE_PER_WAVE]; // Group-sized buffer that contains valid segment indices.
groupshared uint g_RingBufferPrefixScratch[WAVE_COUNT];          // Utility scratch memory for computation of the group prefix sum.
groupshared uint g_RingBufferCount;                              //
groupshared uint g_BatchPos;                                     //

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
// ----------------------------------------
[numthreads(WAVE_COUNT * NUM_LANE_PER_WAVE, 1, 1)]
void RasterBin(uint3 dispatchThreadID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    uint batchPos = 0;

    // Persistent thread:
    // Process the segment data until we exhaust the buffer.
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
        uint bufferIndex = 0;
        uint bufferCount = 0;
        uint batchEnd    = min(batchPos + _BatchSize, _SegmentCount);

        // Keep processing this batch until it is exhausted.
        do
        {
            // Clear the LDS
            g_RingBuffer[groupIndex] = 0;

            // Input phase. Add valid segments into the ring buffer until it is full.
            // This process will compact or expand the results of the segment setup (clipping / tesselation).
            // and ensure that every thread in the group will have some work to do.
            while(bufferCount < WAVE_COUNT * NUM_LANE_PER_WAVE && batchPos < batchEnd)
            {
                const uint segmentIndex = batchPos + groupIndex;

                // Begin in a culled state.
                uint segmentOutput = 0;

                // Determine the number of segments emitted by the segment setup stage / clipper at this index.
                // (Currently we assume zero or one, but potentially more in the future if tessellated).
                if (segmentIndex < batchEnd)
                {
                    segmentOutput = _SegmentOutputBuffer[segmentIndex];
                }

                // Compute the exclusive prefix sum of the segment number across the thread block.
                // Use it to determine an index into the ring buffer.
                const uint ringBufferIndex = bufferCount + GroupPrefixSum(groupIndex, segmentOutput);

                // Update the indices of the loop.
                {
                    // Find the maximum index in the thread block.
                    InterlockedMax(g_RingBufferCount, ringBufferIndex);
                    GroupMemoryBarrierWithGroupSync();

                    // Shift forward the batch position as well
                    g_BatchPos = batchPos + WAVE_COUNT * NUM_LANE_PER_WAVE;
                }

                // Record the segment to the ring buffer.
                if (segmentOutput)
                {
                    g_RingBuffer[ringBufferIndex] = segmentIndex;
                }
                GroupMemoryBarrierWithGroupSync();

                // TEMP: Write the ring buffer in LDS out to global memory to debug.
                _RingBuffer[batchPos + groupIndex] = g_RingBuffer[groupIndex];

                bufferCount = g_RingBufferCount;
                batchPos    = g_BatchPos;
            }

            // TODO: Process the ring buffer. Assign a segment per-lane.

            // Early out for now.
            break;
        }
        while(bufferCount > 0 || batchPos < batchEnd);

        // Rasterization phase.

        return;
    }
}