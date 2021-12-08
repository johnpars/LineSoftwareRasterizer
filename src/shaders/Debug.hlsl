#include "Utility.hlsl"

cbuffer SegmentsPerTileConstants : register(b0)
{
    uint2 GroupDim;
}

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

    _OutputTarget[dispatchThreadID.xy] = OverlayHeatMap(samplePos, uint2(16, 16), tileValue, 70, 1);
}