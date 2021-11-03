cbuffer Constants : register(b0)
{
    float2 Resolution;
    float  Time;
    float  Unused;
}

RWTexture2D<float4> OutputTarget : register(u0);

// Recreation of the ShaderToy template for now.

[numthreads(8, 8, 1)]
void Main(int3 id : SV_DispatchThreadID)
{
    // Compute the normalized coordinate.
    float2 uv = id.xy / Resolution;

    // Time varying color.
    float3 color = 0.5 + 0.5 * cos(Time + uv.xyx + float3(0, 2, 4));

    OutputTarget[id.xy] = float4(color, 1.0);
}