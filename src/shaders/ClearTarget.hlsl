cbuffer Constants : register(b0)
{
    float4 ClearColor;
}

RWTexture2D<float4> OutputTarget : register(u0);

[numthreads(8, 8, 1)]
void Main(uint2 dispatchThreadID : SV_DispatchThreadID)
{
    OutputTarget[dispatchThreadID] = ClearColor;
}