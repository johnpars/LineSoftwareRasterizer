cbuffer Constants : register(b0)
{
    // TODO
}

RWTexture2D<float4> OutputTarget : register(u0);

[numthreads(8, 8, 1)]
void Main(int3 id : SV_DispatchThreadID)
{
    OutputTarget[id.xy] = float4(1.0, 0.0, 0.0, 1.0);
}