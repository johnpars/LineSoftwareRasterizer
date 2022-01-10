#include "RasterCommon.hlsl"

// Inputs
cbuffer Constants : register(b0)
{
    float4 _Params0;
};

ByteAddressBuffer              _HeadPointerBuffer  : register(t0);
StructuredBuffer<FragmentData> _FragmentDataBuffer : register(t1);

// Outputs
RWTexture2D<float4> _OutputTarget : register(u0);

// Defines
#define _ScreenParams _Params0.xy

[numthreads(8, 8, 1)]
void RasterResolve(uint3 dti : SV_DispatchThreadID)
{
    // Index into the 1D screen buffer.
    uint flattenedIndex = dti.x + _ScreenParams.x * dti.y;

    // Load the head pointer for this pixel.
    uint head = _HeadPointerBuffer.Load(4 * flattenedIndex);

    if (head > 1)
        return;

    _OutputTarget[uint2(dti.xy)] = float4(dti.xy / float2(1280, 720), 0, 1);
}