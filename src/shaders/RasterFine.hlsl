#include "RasterCommon.hlsl"
#include "debug/DebugUtility.hlsl"

// Input
cbuffer Constants : register(b0)
{
    float4 _Params0;
    float4 _Params1;
};


Buffer<uint> _WorkQueueBuffer  : register(t0);
Buffer<uint> _BinOffsetBuffer  : register(t1);
Buffer<uint> _BinCounterBuffer : register(t2);

StructuredBuffer<SegmentData>  _SegmentDataBuffer  : register(t3);
StructuredBuffer<VertexOutput> _VertexOutputBuffer : register(t4);

// Output
RWTexture2D<float4> _OutputTarget : register(u0);

// Define
#define _ScreenParams _Params0.xy
#define _TileSize     _Params0.z
#define _TileSizeSS   2.0 * float2(_TileSize.xx / _ScreenParams)
#define _TileDim      uint2(_Params0.w, _Params1.x)
#define _CurveSamples _Params1.y

// Local
groupshared uint g_BinOffset;
groupshared uint g_BinCount;

// Kernel
[numthreads(16, 16, 1)]
void RasterFine(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    // Convert the dispatch coordinates to NDC.
    const float2 UV = ((float2)dispatchThreadID.xy + 0.5) * rcp(_ScreenParams);
    const float2 UVh = -1 + 2 * UV;

    const float segmentWidth = 1.5 / _ScreenParams.y;

    // Load the tile data into LDS.
    if (groupIndex == 0)
    {
        const uint binIndex = groupID.x + _TileDim.x * groupID.y;
        g_BinOffset = _BinOffsetBuffer[binIndex];
        g_BinCount  = _BinCounterBuffer[binIndex];
    }
    GroupMemoryBarrierWithGroupSync();

    uint segmentCount = g_BinCount;
    uint binOffset    = g_BinOffset;

    float3 result = 0;

    float Z = -FLT_MAX;

    for (uint s = 0; s < segmentCount; ++s)
    {
        // Load the segment index.
        uint segmentIndex = _WorkQueueBuffer[binOffset + s];

        // Load the segment indices.
        SegmentData data = _SegmentDataBuffer[segmentIndex];

        // Load Vertex Data
        const VertexOutput v0 = _VertexOutputBuffer[data.vi0];
        const VertexOutput v1 = _VertexOutputBuffer[data.vi1];

        // We want the barycentric between the original segment vertices, not the clipped vertices.
        float3 p0 = v0.positionCS.xyz * rcp(v0.positionCS.w);
        float3 p1 = v1.positionCS.xyz * rcp(v1.positionCS.w);

        // Compute the segment coverage and 'barycentric' coord.
#ifndef RASTER_CURVE
        float t;
        float distance = DistanceToSegmentAndTValue(UVh, p0.xy, p1.xy, t);
#else
        float2 controlPoints[4];
        LoadControlPoints(segmentIndex, _SegmentDataBuffer, _VertexOutputBuffer, controlPoints);

        float t;
        float distance = DistanceToCubicBezierAndTValue(UVh, controlPoints, t, _CurveSamples);
#endif

        // Compute the segment coverage provided by the segment distance.
        float coverage = 1 - smoothstep(0.0, segmentWidth, distance);

        float2 coords = float2(
            t,
            1 - t
        );

        // Interpolate Vertex Data
        const float z  = INTERP(coords, p0.z, p1.z);
        const float texCoord = INTERP(coords, v0.texCoord, v1.texCoord);

        if (coverage && z > Z)
        {
            result = lerp(float3(1, 0, 1), float3(0, 1, 1), texCoord) * coverage;
            // result = ColorCycle(segmentIndex % 12, 12) * coverage;
            Z = z;
        }
    }

    _OutputTarget[dispatchThreadID.xy] = float4(result, 1);
}