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
#define _Opacity        _Params1.z
#define _HeatmapOverlay _Params1.w

// Hardcoded for 32 slices due to the slice mask.
#define NUM_SLICES 128

// Static Global
// Warning, slice mask can only support up to 128 slices.
static uint4 s_SliceMask;

// Local
groupshared uint g_BinOffset;
groupshared uint g_BinCount;
groupshared uint g_BinMinZ;
groupshared uint g_BinMaxZ;

// Utility

uint GetLeastSignificantBit(uint mask)
{
    // Ref: https://graphics.stanford.edu/~seander/bithacks.html
    static const int MultiplyDeBruijnBitPosition[32] =
    {
          0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
          31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
    };

    // Find the least significant set bit in the mask.
    return MultiplyDeBruijnBitPosition[(((mask & -mask) * 0x077CB531U)) >> 27];
}

uint ComputeSliceIndex(float binStart, float binEnd, float z)
{
    const float fraction = (z - binStart) * rcp(binEnd - binStart);
    return clamp(fraction * NUM_SLICES, 0, NUM_SLICES - 1);
}

bool IsSliceEmpty(uint sliceIndex)
{
    uint mask;

    if      (sliceIndex < 32) { mask = s_SliceMask.x; }
    else if (sliceIndex < 64) { mask = s_SliceMask.y; }
    else if (sliceIndex < 96) { mask = s_SliceMask.z; }
    else                      { mask = s_SliceMask.w; }

    return (mask & (1u << sliceIndex)) == 0;
}

void WriteSlice(uint sliceIndex)
{
    if      (sliceIndex < 32) { s_SliceMask.x |= 1u << sliceIndex; }
    else if (sliceIndex < 64) { s_SliceMask.y |= 1u << sliceIndex; }
    else if (sliceIndex < 96) { s_SliceMask.z |= 1u << sliceIndex; }
    else                      { s_SliceMask.w |= 1u << sliceIndex; }
}

// Kernel
[numthreads(16, 16, 1)]
void RasterFineOIT(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    // Convert the dispatch coordinates to NDC.
    const float2 UV = ((float2)dispatchThreadID.xy + 0.5) * rcp(_ScreenParams);
    const float2 UVh = -1 + 2 * UV;

    const float segmentWidth = 2  / _ScreenParams.y;

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

    const float binMinZ = asfloat(g_BinMinZ);
    const float binMaxZ = asfloat(g_BinMaxZ);

    if (segmentCount == 0)
        return;

    // Slice and fragment buffers.
    uint   slices    [NUM_SLICES];
    float4 fragments [NUM_SLICES];

    // Maintain a bit mask to check for slice buffer occupants.
    s_SliceMask = 0;

    // Track a fragment counter for new entries to the fragment buffer.
    uint fragmentCounter = 0;

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

        // Skip the segment if there is no coverage.
        if (!coverage)
            continue;

        coverage *= _Opacity;

        float2 coords = float2(
            t,
            1 - t
        );

        // Interpolate vertex data.
        const float z = INTERP(coords, p0.z, p1.z);
        const float texCoord = INTERP(coords, v0.texCoord, v1.texCoord);

        // Invoke fragment shader / sample and blend offscreen shading.
        // float4 fragment = float4(ColorCycle(floor(segmentIndex / 9), 2) * coverage, coverage);
        float4 fragment = float4(lerp(float3(1, 0, 1), float3(0, 1, 1), texCoord) * coverage, coverage);

        // Compute the slice index for this depth value.
        const uint sliceIndex = ComputeSliceIndex(binMaxZ, binMinZ, z);

        if (!IsSliceEmpty(sliceIndex))
        {
            // This slice is already occupied.
            const uint sliceFragmentIndex = slices[sliceIndex];

            float4 sliceFragment = fragments[sliceFragmentIndex];
            {
                // Alpha blend with the current fragment in the slice.
                sliceFragment += (fragment * (1 - sliceFragment.a));
            }

            fragments[sliceFragmentIndex] = sliceFragment;

            // Proceed with the next segment.
            continue;
        }

        // First update the slice mask.
        WriteSlice(sliceIndex);

        // Make the new fragment entry.
        fragments[fragmentCounter] = fragment;

        // Point to it in the slice buffer.
        slices[sliceIndex] = fragmentCounter;

        // Iterate the counter for future entries.
        fragmentCounter++;
    }

    if (fragmentCounter == 0)
        return;

    // float4 pixelColorAndAlpha = fragments[slices[GetLeastSignificantBit(sliceMask)]]; //float4(0, 0, 0, 1);
    float4 pixelColorAndAlpha = float4(0, 0, 0, 1);

    // Scan the slices in order to resolve the per-pixel transmittance function.
    uint f, i = 0;
    for (; i < NUM_SLICES; ++i)
    {
        // Skip empty slices.
        if (IsSliceEmpty(i))
            continue;

        // Fetch the slice pointer to the fragment.
        const int fragmentIndex = slices[i];

        // Grab the fragment
        const float4 fragmentColorAndAlpha = fragments[fragmentIndex];

        // Ordered transmittance function.
        pixelColorAndAlpha.rgb += fragmentColorAndAlpha.rgb * pixelColorAndAlpha.a;
        pixelColorAndAlpha.a   *= 1 - fragmentColorAndAlpha.a;

        // Only iterate if this was a non-empty slice.
        // f++;
    }

    float4 base = pixelColorAndAlpha;
    float4 heat = OverlayHeatMap(dispatchThreadID.xy, uint2(0, 0), fragmentCounter, 32, 1.0);

    const float a = _HeatmapOverlay;
    _OutputTarget[dispatchThreadID.xy] = float4((base.rgb * (1 - a)) + (heat.rgb * a), 1);
}