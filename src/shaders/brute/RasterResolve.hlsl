#include "RasterCommon.hlsl"

struct Fragment
{
    float3 a;
    float  t;
    float  z;
};

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
#define _ScreenParams _Params0.yz

#define LAYERS_PER_PIXEL 16
#define LAST_NODE        0xFFFFFFFF

void InitializeBlendingArray(inout Fragment B[LAYERS_PER_PIXEL + 1])
{
    // Create the default fragment.
    // (These defaults are important due to how we merge for memory compression).
    Fragment F;
    F.a = 0;
    F.t = 1;
    F.z = -FLT_MAX;

    for (int i = 0; i < LAYERS_PER_PIXEL + 1; i++)
    {
        B[i] = F;
    }
}

// Implementation of "Multi-Layer Alpha Blending"
// Ref: https://www.intel.com/content/dam/develop/external/us/en/documents/i3d14-mlab-preprint.pdf
void InsertFragment(in Fragment F, inout Fragment B[LAYERS_PER_PIXEL + 1])
{
    // 1-Pass bubble sort to insert the fragment.
    Fragment temp, merge;
    for (int i = 0; i < LAYERS_PER_PIXEL + 1; i++)
    {
        if (F.z >= B[i].z)
        {
            temp = B[i];
            B[i] = F;
            F    = temp;
        }
    }

    // Compression (merge the last two rows since we have a fixed memory size).
    const int m = LAYERS_PER_PIXEL;
    merge.a = B[m - 1].a + B[m].a * B[m - 1].t;
    merge.t = B[m - 1].t * B[m].t;
    merge.z = B[m - 1].z;
    B[m - 1] = merge;
}

[numthreads(8, 8, 1)]
void RasterResolve(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Index into the 1D screen buffer.
    uint flattenedIndex = dispatchThreadID.x + (_ScreenParams.x * dispatchThreadID.y);

    // Load the head pointer for this pixel.
    uint next = _HeadPointerBuffer.Load(4 * flattenedIndex);

    // Create and initialize the blending array.
    Fragment B[LAYERS_PER_PIXEL + 1];
    InitializeBlendingArray(B);

    float4 result = 0;

    // Walk down the per-pixel fragment linked list and sort the blending array.
    while (next != LAST_NODE)
    {
         const FragmentData node = _FragmentDataBuffer[next];
         {
            Fragment f;
            f.a = node.color.rgb * node.color.a;
            f.t = 1.0 - node.color.a;
            f.z = node.depth;
            InsertFragment(f, B);
         }
         next = node.next;
    }

    float transmittance = 1;

    // Resolve the per-pixel transmittance function.
    for (int k = 0; k < LAYERS_PER_PIXEL + 1; k++)
    {
        result.rgb = result.rgb + transmittance * B[k].a;
        transmittance *= B[k].t;
    }

    _OutputTarget[dispatchThreadID.xy] = result;
}