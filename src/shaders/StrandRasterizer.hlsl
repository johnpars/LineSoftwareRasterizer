struct Vertex
{
    float3 positionOS;
};

StructuredBuffer<Vertex> _VertexBuffer : register(t0);
Buffer<int>              _IndexBuffer  : register(t1);

cbuffer Constants : register(b0)
{
    float4x4 _MatrixV;
    float4x4 _MatrixP;
    float4   _ScreenParams;
    float4   _Params0;
}

#define _SegmentCount _Params0.x

RWTexture2D<float4> _OutputTarget : register(u0);

float Line(float2 P, float2 A, float2 B)
{
    float2 AB = B - A;
    float2 AP = P - A;

    float2 T = normalize(AB);
    float l = length(AB);

    float t = clamp(dot(T, AP), 0.0, l);
    float2 closestPoint = A + t * T;

    float distanceToClosest = 1.0 - (length(closestPoint - P) / 0.005);
    float i = clamp(distanceToClosest, 0.0, 1.0);

    return sqrt(i);
}

[numthreads(8, 8, 1)]
void Main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Convert the dispatch coordinates to the generation space [0, 1]
    const float2 UV = ((float2)dispatchThreadID.xy + 0.5) * _ScreenParams.zw;
    const float2 UVh = -1 + 2 * UV;

    float3 result = _OutputTarget[dispatchThreadID.xy].xyz;

    for (int i = 0; i < _SegmentCount; ++i)
    {
        // Load Indices
        const int i0 = _IndexBuffer[i + 0];
        const int i1 = _IndexBuffer[i + 1];

        // Load Vertices
        const Vertex v0 = _VertexBuffer[i0];
        const Vertex v1 = _VertexBuffer[i1];

        // Project to screen
        const float4 h0 = mul(mul(float4(v0.positionOS, 1.0), _MatrixV), _MatrixP);
        const float4 h1 = mul(mul(float4(v1.positionOS, 1.0), _MatrixV), _MatrixP);

        // Perspective divide
        const float3 p0 = h0.xyz / h0.w;
        const float3 p1 = h1.xyz / h1.w;

        // Accumulate Result
        result = max(result, Line(UVh, p0.xy, p1.xy));
    }

    _OutputTarget[dispatchThreadID.xy] = float4(result, 1.0);
}
