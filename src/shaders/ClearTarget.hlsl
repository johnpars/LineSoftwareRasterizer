cbuffer ClearTargetConstants : register(b0)
{
    float4 ClearColor;
}

RWTexture2D<float4> _OutputTarget : register(u0);

[numthreads(8, 8, 1)]
void ClearTarget(uint2 dispatchThreadID : SV_DispatchThreadID)
{
    _OutputTarget[dispatchThreadID] = ClearColor;
}

cbuffer ClearBufferConstants : register(b0)
{
    uint Value;
    uint Count;
}

RWBuffer<uint> _OutputBuffer : register(u0);

[numthreads(16, 1, 1)]
void ClearBuffer(uint2 dispatchThreadID : SV_DispatchThreadID,
                 uint3 groupID : SV_GroupID)
{
    if (dispatchThreadID.x >= Count)
        return;

    _OutputBuffer[dispatchThreadID.x] = Value;
}