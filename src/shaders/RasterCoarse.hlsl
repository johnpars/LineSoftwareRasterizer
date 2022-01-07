#include "RasterCommon.hlsl"

// Inputs
// ----------------------------------------
StructuredBuffer<SegmentData> _SegmentBuffer : register(t0);

// Outputs
// ----------------------------------------
RWBuffer<uint> _CounterBuffer : register(u0);

// Define
// ----------------------------------------
#define NUM_WARP 8

// Kernel
// ----------------------------------------
[numthreads(NUM_WARP * NUM_THREAD_PER_WARP, 1, 1)]
void RasterCoarse(uint3 dispatchThreadID : SV_DispatchThreadID)
{
}