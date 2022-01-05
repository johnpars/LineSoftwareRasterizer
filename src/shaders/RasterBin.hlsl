#include "Common.hlsl"

// Inputs
// ----------------------------------------
cbuffer ConstantsSetup : register(b0)
{
    float4 _Params;
}

// TODO: If we end up not doing tesselation in the segment setup, we can go from uint -> bool. We should also switch
// to ByteAddressBuffer regardless.
StructuredBuffer<uint> _SegmentCountBuffer : register(t0);

// Outputs
// ----------------------------------------
RWBuffer<uint> _CounterBuffer : register(u0);

// Define
// ----------------------------------------
#define _SegmentCount _Params.x
#define _BatchSize    _Params.y

// Launch 512 threads per processor.
#define NUM_WARP 8

// Local
// ----------------------------------------
groupshared uint g_RingBuffer[NUM_WARP * NUM_THREAD_PER_WARP];  //
groupshared uint g_RingBufferPrefixScratch[NUM_WARP];           // Utility buffer for computation of the prefix sum.
groupshared uint g_BatchPos;                                    //

// Utility
// ----------------------------------------

// Perform a group-wide exclusive prefix sum. Returns the prefix sum of x's register at the current lane.
// TODO: There is likely much better ways to do this, start with this approach for now and optimize later.
uint GroupPrefixSum(uint laneID, uint x)
{
    uint waveID = WaveIndex(laneID);

    // Just allow all the lanes to assign the wave sum to the LDS, not exactly sure if this is 'better' than having the
    // first lane in the wave assign it, but if we do that then you need to be careful of computing the wave sum beforehand
    // (else-wise all lanes aside from the first one will be inactive and the sum result be invalid). Since it's stored in
    // and SGPR anyway this is probably ok for now.
    g_RingBufferPrefixScratch[waveID] = WaveActiveSum(x);

    // Allow the first lane in the block to compute the prefix sum on the wave sized list.
    if (laneID == 0)
    {
        uint i;

        // 1) Compute the inclusive prefix sum.
        for (i = 1; i < NUM_WARP; ++i)
        {
            g_RingBufferPrefixScratch[i] += g_RingBufferPrefixScratch[i - 1];
        }

        // 2) Shift elements once to the right.
        for (i = NUM_WARP; i >= 1; --i)
        {
            g_RingBufferPrefixScratch[i] = g_RingBufferPrefixScratch[i - 1];
        }

        // 3) Replace the first element with the identity to arrive at the exclusive sum.
        g_RingBufferPrefixScratch[0] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    return g_RingBufferPrefixScratch[waveID] + WavePrefixSum(x);
}


// Kernel (1 Thread Block = 1 SM/CU)
// ----------------------------------------
[numthreads(NUM_WARP * NUM_THREAD_PER_WARP, 1, 1)]
void RasterBin(uint3 dispatchThreadID : SV_DispatchThreadID,
               uint  groupIndex : SV_GroupIndex)
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

        // Input phase. Process the batch into the ring buffer as long as there are segments in it.
        // This process will compact or expand the results of the segment setup (clipping / tesselation).
        // and ensure that every thread in the group will have some work to do.
        do
        {
            // Keep reading segments in the batch until we have group-size amount of valid segments in the ring buffer.
            while(bufferCount < NUM_WARP * NUM_THREAD_PER_WARP && batchPos < batchEnd)
            {
                uint segmentIndex = batchPos + groupIndex;

                uint num;

                // Determine the number of segments emitted by the segment setup stage / clipper at this index.
                // (Currently we assume zero or one, but potentially more in the future if tessellated).
                if (segmentIndex < batchEnd)
                {
                    num = _SegmentCountBuffer[segmentIndex];
                }

                // TODO: If all have work, just do a fast path that has 1-1 mapping of batch -> ring buffer.

                // See if it brings any additional perf to the binning.
                uint ringBufferIndex = GroupPrefixSum(groupIndex, num);

                // Record the segment to the ring buffer.
                if (num)
                {
                    g_RingBuffer[ringBufferIndex] = segmentIndex;
                }
                GroupMemoryBarrierWithGroupSync();

                // Early out for now.
                return;
            }

            // Early out for now.
            return;
        }
        while(bufferCount > 0 || batchPos < batchEnd);

        // Rasterization phase.
    }
}