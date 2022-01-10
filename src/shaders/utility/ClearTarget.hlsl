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