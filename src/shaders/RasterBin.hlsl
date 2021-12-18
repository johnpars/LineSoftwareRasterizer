#include "Common.hlsl"

// Inputs
// ----------------------------------------
cbuffer ConstantsSetup : register(b0)
{
    float4 _Params;
}

StructuredBuffer<SegmentData> _SegmentBuffer : register(t0);

// Outputs
// ----------------------------------------
RWBuffer<uint> _CounterBuffer : register(u0);

// Define
// ----------------------------------------
#define _SegmentCount _Params.x
#define _BatchSize    _Params.y

#define NUM_WARP   16

// Local
// ----------------------------------------
groupshared uint g_SegmentRingBuffer[NUM_WARP * NUM_THREAD_PER_WARP];
groupshared uint g_BatchPos;

// Kernel (1 Thread Block = 1 SM)
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

        // Input phase. Process the batch as long as there are segments in it.
        do
        {
            // Keep reading segments in the batch until we have GroupSize amount of segments in the ring buffer.
            while(bufferCount < NUM_WARP * NUM_THREAD_PER_WARP && batchPos < batchEnd)
            {
                int segmentIndex = batchPos + groupIndex;

                return;
            }
        }
        while(bufferCount > 0 || batchPos < batchEnd);

        // Rasterization phase.
    }
}