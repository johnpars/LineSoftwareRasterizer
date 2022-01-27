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

Buffer<uint> _BinMinZ          : register(t5);
Buffer<uint> _BinMaxZ          : register(t6);

// Output
RWTexture2D<float4> _OutputTarget : register(u0);

// Define
#define _ScreenParams _Params0.xy
#define _TileSize     _Params0.z
#define _TileSizeSS   2.0 * float2(_TileSize.xx / _ScreenParams)
#define _TileDim      uint2(_Params0.w, _Params1.x)
#define _CurveSamples _Params1.y

#define NUM_SLICES 64

// Local
groupshared uint g_BinOffset;
groupshared uint g_BinCount;
groupshared uint g_BinMinZ;
groupshared uint g_BinMaxZ;

// Utility

uint ComputeSliceIndex(float binStart, float binEnd, float z)
{
    const float fraction = (z - binStart) / (binEnd - binStart);
    return clamp(round(fraction * NUM_SLICES), 0, 64);
}

// Kernel
[numthreads(16, 16, 1)]
void RasterFineOIT(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    // Convert the dispatch coordinates to NDC.
    const float2 UV = ((float2)dispatchThreadID.xy + 0.5) * rcp(_ScreenParams);
    const float2 UVh = -1 + 2 * UV;

    const float segmentWidth = 3.0 / _ScreenParams.y;

    // Load the tile data into LDS.
    if (groupIndex == 0)
    {
        const uint binIndex = groupID.x + _TileDim.x * groupID.y;
        g_BinOffset = _BinOffsetBuffer[binIndex];
        g_BinCount  = _BinCounterBuffer[binIndex];
        g_BinMinZ   = _BinMinZ[binIndex];
        g_BinMaxZ   = _BinMaxZ[binIndex];
    }
    GroupMemoryBarrierWithGroupSync();

    const uint segmentCount = g_BinCount;
    const uint binOffset    = g_BinOffset;

    const float binMinZ      = asfloat(g_BinMinZ);
    const float binMaxZ      = asfloat(g_BinMaxZ);

    int sliceBuffer[NUM_SLICES];

    for (uint k = 0; k < NUM_SLICES; k++)
        sliceBuffer[k] = -1;

    uint fragmentCounter = 0;
    float4 fragments[NUM_SLICES];

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
        float t;
        float distance = DistanceToSegmentAndTValue(UVh, p0.xy, p1.xy, t);

        // Compute the segment coverage provided by the segment distance.
        float coverage = 1 - smoothstep(0.0, segmentWidth, distance);

        // Skip the segment if there is no coverage.
        if (!coverage)
            continue;

        float2 coords = float2(
            t,
            1 - t
        );

        // Interpolate Vertex Data
        const float z = INTERP(coords, p0.z, p1.z);

        // Compute the slice index for this depth value and point to the fragment index.
        const uint sliceIndex = ComputeSliceIndex(binMaxZ, binMinZ, z);

        float4 fragment = float4(ColorCycle(floor(segmentIndex / 10), 100) * coverage, coverage);

        if (sliceBuffer[sliceIndex] < 0)
        {
            fragments[fragmentCounter] = fragment;
            sliceBuffer[sliceIndex] = fragmentCounter;
            fragmentCounter++;
        }
        else
        {
            // This slice is already occupied. Alpha blend with the current fragment.
            float4 slicFragment = fragments[sliceBuffer[sliceIndex]];
            float4 blenFragment = slicFragment + (fragment * (1 - slicFragment.a));
            fragments[sliceBuffer[sliceIndex]] = blenFragment;
        }
    }

    float3 result = 0;
    float transmittance = 1;

    // Resolve the per-pixel transmittance function.
    for (int i = 0; i < fragmentCounter; i++)
    {
        const int fragmentIndex = sliceBuffer[i];

        // Skip empty slices.
        if (fragmentIndex < 0)
            continue;

        const float4 color = fragments[fragmentIndex];

        result.rgb = result.rgb + transmittance * color.rgb;
        transmittance *= 1 - color.a;
    }

    _OutputTarget[dispatchThreadID.xy] = float4(result, transmittance);
}