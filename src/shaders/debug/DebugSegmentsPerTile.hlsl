#include "debug/DebugUtility.hlsl"

cbuffer SegmentsPerTileConstants : register(b0)
{
    uint2 GroupDim;
    float Opacity;
};

Buffer<uint>        _TileSegmentCountBuffer : register(t1);
RWTexture2D<float4> _OutputTarget           : register(u0);

[numthreads(16, 16, 1)]
void SegmentsPerTile(uint3 dispatchThreadID : SV_DispatchThreadID,
                     uint3 groupID          : SV_GroupID)
{
    uint tileValue = _TileSegmentCountBuffer[(groupID.y * GroupDim.x) + groupID.x];

    uint2 samplePos = dispatchThreadID.xy;
    {
        // Flip the Y
        samplePos.y = -samplePos.y;
    }

    float4 base = _OutputTarget[dispatchThreadID.xy];
    float4 heat = OverlayHeatMap(samplePos, uint2(16, 16), tileValue, 20000, 1.0);

    const float a = Opacity;
    _OutputTarget[dispatchThreadID.xy] = float4((base.rgb * (1 - a)) + (heat.rgb * a), 1);
}