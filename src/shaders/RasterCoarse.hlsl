#include "Common.hlsl"

// Inputs
// ----------------------------------------
StructuredBuffer<SegmentData> _SegmentBuffer : register(t0);

// Outputs
// ----------------------------------------
RWBuffer<uint> _CounterBuffer : register(u0);

// Kernel
// ----------------------------------------
[numthreads(NUM_WARP_PER_SM * NUM_THREAD_PER_WARP, 1, 1)]
void RasterCoarse(uint3 dispatchThreadID : SV_DispatchThreadID)
{
}