cbuffer ClearBufferConstants : register(b0)
{
    uint Value;
    uint Count;
}

RWBuffer<uint> _OutputBuffer : register(u0);

[numthreads(64, 1, 1)]
void ClearBuffer(uint2 dispatchThreadID : SV_DispatchThreadID)
{
    const uint i = dispatchThreadID.x;

    if (i >= Count)
        return;

    _OutputBuffer[i] = Value;
}