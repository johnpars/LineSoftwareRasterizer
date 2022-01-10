#include "RasterCommon.hlsl"

// Inputs
// ----------------------------------------
StructuredBuffer<SegmentData> _SegmentBuffer : register(t0);

// Outputs
// ----------------------------------------
RWBuffer<uint> _CounterBuffer : register(u0);

// Define
// ----------------------------------------
#define NUM_WAVE 8

// Kernel
// ----------------------------------------
[numthreads(NUM_WAVE * NUM_LANE_PER_WAVE, 1, 1)]
void RasterCoarse(uint3 dispatchThreadID : SV_DispatchThreadID)
{
}