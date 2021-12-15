#include "Common.hlsl"
#include "Vert.hlsl"

// Inputs

cbuffer ConstantsSetup : register(b0)
{
    float4 _Params;
}

StructuredBuffer<VertexInput> _VertexBuffer     : register(t0);
Buffer<uint>                  _IndexBuffer      : register(t1);

// Outputs
RWStructuredBuffer<SegmentData> _SegmentBuffer : register(u0);

// Defines
#define _SegmentCount _Params.x

// Util
bool ClipVertex(VertexOutput v)
{
    return false;
}

// Kernel
[numthreads(GROUP_SIZE_1D, 1, 1)]
void SegmentSetup(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint i = dispatchThreadID.x;

    if (i >= _SegmentCount)
        return;

    // Load Indices
    const uint i0 = _IndexBuffer[i * 2 + 0];
    const uint i1 = _IndexBuffer[i * 2 + 1];

    // Load Vertices
    const VertexInput i_v0 = _VertexBuffer[i0];
    const VertexInput i_v1 = _VertexBuffer[i1];

    // Invoke Vertex Shader
    VertexOutput o_v0 = Vert(i_v0);
    VertexOutput o_v1 = Vert(i_v1);

    // Clip if behind the projection plane.
    if (ClipVertex(o_v0) || ClipVertex(o_v1))
        return;

    // Perspective divide
    o_v0.positionCS.xyz = o_v0.positionCS.xyz / o_v0.positionCS.w;
    o_v1.positionCS.xyz = o_v1.positionCS.xyz / o_v1.positionCS.w;

    // Write back to the segment buffer
    SegmentData segment;
    {
        segment.v0 = o_v0;
        segment.v1 = o_v1;
    }
    _SegmentBuffer[i] = segment;
}