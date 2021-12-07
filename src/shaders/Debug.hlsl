#include "Utility.hlsl"

RWTexture2D<float4> _OutputTarget : register(u0);

float3 ColorCycle(uint index, uint count)
{
	float t = frac(index / (float)count);

	// Ref: https://www.shadertoy.com/view/4ttfRn
	float3 c = 3.0 * float3(abs(t - 0.5), t.xx) - float3(1.5, 1.0, 2.0);
	return 1.0 - c * c;
}

[numthreads(16, 16, 1)]
void SegmentsPerTile(uint3 dispatchThreadID : SV_DispatchThreadID,
                     uint3 groupThreadID : SV_GroupID)
{
    uint2 samplePos = dispatchThreadID.xy;
    samplePos.y = -samplePos.y;
    _OutputTarget[dispatchThreadID.xy] = OverlayHeatMap(samplePos, uint2(16, 16), groupThreadID.x, 256, 1);
}