cbuffer ClearBufferConstants : register(b0)
{
    uint Value;
    uint Count;
}

RWBuffer<int> _OutputBuffer : register(u0);

[numthreads(16, 1, 1)]
void ClearBuffer(uint2 dispatchThreadID : SV_DispatchThreadID,
                 uint3 groupID : SV_GroupID)
{
    if (dispatchThreadID.x >= Count)
        return;

    _OutputBuffer[dispatchThreadID.x] = Value;
}