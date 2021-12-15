#include "Utility.hlsl"

// Structures
// -----------------------------------------------------
struct StrandData
{
    float3 strandPositionOS;
};

struct Vertex
{
    float vertexID;
    float vertexUV;
};

struct Fragment
{
    float3 a;
    float  t;
    float  z;
};

struct VertexOutput
{
    float2 texcoord;
    float3 positionOS;
    float4 positionCS;
};

struct FragmentInput
{
};

// Buffers
// -----------------------------------------------------
StructuredBuffer<Vertex>     _VertexBuffer     : register(t0);
Buffer<uint>                 _IndexBuffer      : register(t1);
StructuredBuffer<StrandData> _StrandDataBuffer : register(t2);
RWTexture2D<float4>          _OutputTarget     : register(u0);

cbuffer Constants : register(b0)
{
    float4x4 _MatrixV;
    float4x4 _MatrixP;
    float4   _ScreenParams;
    float4   _Params0;
}

// Defines
// -----------------------------------------------------
#define INTERP(coords, a, b) coords.y * a + coords.x * b
#define ZERO_INITIALIZE(type, name) name = (type)0;

// #define LAYOUT_INTERLEAVED 1 // Force to interleaved for now - however for OBJ needs to be sequential
#define DEBUG_COLOR_STRAND 1
//#define OPAQUE 1
#define PERSPECTIVE_CORRECT_INTERPOLATION 1

// Maximum representable floating-point number
#define FLT_MAX  3.402823466e+38

#define _StrandCount           _Params0.x
#define _StrandParticleCount   _Params0.y
#define _PerStrandVertexCount  _StrandParticleCount
#define _PerStrandSegmentCount (_PerStrandVertexCount - 1)
#define _PerStrandIndexCount   (_PerStrandSegmentCount * 2)
#define _IndexCount            _StrandCount * _PerStrandIndexCount

// The maximum number of fragments to correctly blend per-pixel.
#define LAYERS_PER_PIXEL 1

#if LAYOUT_INTERLEAVED
    #define DECLARE_STRAND(x)							\
        const uint strandIndex = x;						\
        const uint strandParticleBegin = strandIndex;	\
        const uint strandParticleStride = _StrandCount;	\
        const uint strandParticleEnd = strandParticleBegin + strandParticleStride * _StrandParticleCount;
#else
    #define DECLARE_STRAND(x)													\
        const uint strandIndex = x;												\
        const uint strandParticleBegin = strandIndex * _StrandParticleCount;	\
        const uint strandParticleStride = 1;									\
        const uint strandParticleEnd = strandParticleBegin + strandParticleStride * _StrandParticleCount;
#endif

#if DEBUG_COLOR_STRAND
    #define DEBUG_COLOR(x) ColorCycle(x, _StrandCount)
#else
    #define DEBUG_COLOR(x) ColorCycle(x, _StrandCount * _StrandParticleCount)
#endif

// Signed distance to a line segment.
// Ref: https://www.shadertoy.com/view/3tdSDj
float ComputeSegmentCoverageAndBarycentericCoordinate(float2 P, float2 A, float2 B, out float H)
{
    float2 BA = B - A;
    float2 PA = P - A;

    // Also output the 'barycentric' segment coordinate computed as a bi-product of the coverage.
    H = clamp( dot(PA, BA) / dot(BA, BA), 0.0, 1.0);

    return length(PA - H * BA);
}

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

// Theoretical Vertex Shader Program
float4 Vert(uint vertexID, out float3 positionOS)
{
    uint linearParticleIndex = vertexID;
    DECLARE_STRAND(vertexID / _StrandParticleCount)

    const uint i = strandParticleBegin + (linearParticleIndex % _StrandParticleCount) * strandParticleStride;

    const StrandData strandData = _StrandDataBuffer[i];

    positionOS = strandData.strandPositionOS;

    return mul(mul(float4(strandData.strandPositionOS, 1.0), _MatrixV), _MatrixP);
}

float rand(float co) { return frac(sin(co*(91.3458)) * 47453.5453); }

// Theoretical Fragment Shader Program
float4 Frag(uint strandIndex, float2 uv, float3 positionOS)
{
    const float3 c = DEBUG_COLOR(strandIndex);
    return float4(c, 1); // float4(lerp(c, 1 - c, uv.x), 0.4); // float4(4 * pow(0.5 * positionOS + 0.5, 3), 0.3); // lerp(c, 1 - c, uv.x);
}

[numthreads(8, 8, 1)]
void BruteForce(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Convert the dispatch coordinates to the generation space [0, 1]
    const float2 UV = ((float2)dispatchThreadID.xy + 0.5) * _ScreenParams.zw;
    const float2 UVh = -1 + 2 * UV;

    float3 result = _OutputTarget[dispatchThreadID.xy].xyz;

#if OPAQUE
    // Maintain a Z-Buffer per-thread
    float Z = -FLT_MAX;
#else
    // Create and initialize the blending array.
    Fragment B[LAYERS_PER_PIXEL + 1];
    InitializeBlendingArray(B);
#endif

    // Brute force iterate over every segment.
    // Emulates the rasterization of line strips.
    for (uint i = 0; i < _IndexCount; i += 2)
    {
         // Load Indices
         const uint i0 = _IndexBuffer[i + 0];
         const uint i1 = _IndexBuffer[i + 1];

         // Load Vertices
         const Vertex v0 = _VertexBuffer[i0];
         const Vertex v1 = _VertexBuffer[i1];

         // Invoke Vertex Shader
         float3 positionOS0, positionOS1;
         const float4 h0 = Vert(uint(v0.vertexID), positionOS0);
         const float4 h1 = Vert(uint(v1.vertexID), positionOS1);

         // Clip if behind the projection plane.
         if (h0.z > 0 || h1.z > 0)
            continue;

         // Perspective divide
         float3 p0 = h0.xyz / h0.w;
         float3 p1 = h1.xyz / h1.w;

         // Compute the "barycenteric" coordinate on the segment.
         float coord;
         float d = ComputeSegmentCoverageAndBarycentericCoordinate(UVh, p0.xy, p1.xy, coord);

         // Compute the segment coverage provided by the segment distance.
         float coverage = 1 - smoothstep(0.0, 0.001, d);

         float2 coords = float2(
            coord,
            1 - coord
         );

#if PERSPECTIVE_CORRECT_INTERPOLATION
         // Ensure perspective correct coordinates.
         const float2 w = rcp(float2(h0.w, h1.w));
         coords = (coords * w) / dot(coords, w);
#endif

         // Interpolate Vertex Data
         const float2 uv         = INTERP(coords, v0.vertexUV, v1.vertexUV);
         const float3 positionOS = INTERP(coords, positionOS0, positionOS1);
         const float  z          = INTERP(coords, p0.z,        p1.z);

         // Invoke Fragment Shader and mask by Coverage
         // TODO: Clean up the mess
#if OPAQUE
         if (coverage > 0 && z > Z)
#else
         if (any(coverage))
#endif
         {
#if OPAQUE
            result = Frag(i / _PerStrandIndexCount, uv, positionOS).rgb * coverage;
            Z = z;
#else
            float4 fragmentValue = Frag(i / _PerStrandIndexCount, uv, positionOS);

            float alpha = coverage * fragmentValue.a;

            Fragment f;
            f.a = fragmentValue.rgb * alpha;
            f.t = 1.0 - alpha;
            f.z = z;

            InsertFragment(f, B);
#endif
         }
    }

    // Composite the blending array.
#if !OPAQUE
    result = 0;

    float transmittance = 1;

    for (int k = 0; k < LAYERS_PER_PIXEL + 1; k++)
    {
        result = result + transmittance * B[k].a;
        transmittance *= B[k].t;
    }
#endif

    _OutputTarget[dispatchThreadID.xy] = float4(result, 1.0);
}

// ----------------------------------------------------------------------------

struct AABB
{
    float3 min;
    float3 max;

    float3 Center()
    {
        return (min + max) * 0.5;
    }
};

struct SegmentData
{
    float4 P0;
    float4 P1;
    uint   index;
};

struct TileListNode
{
    SegmentData data;
    uint        next;
};

RWBuffer<uint>                   _SegmentCountBuffer     : register(u0);
RWByteAddressBuffer              _TileHeadPointerBuffer  : register(u1);
RWStructuredBuffer<TileListNode> _TileSegmentDataBuffer  : register(u2);
RWStructuredBuffer<uint>         _TileSegmentDataCounter : register(u3);

#define _GroupDim     _Params0.xy
#define _TileSize     _Params0.z
#define _TileSizeSS   2.0 * float2(_TileSize.xx / _ScreenParams.xy)
#define _SegmentCount _Params0.w

bool SegmentIntersectsTile(float3 p0, float3 p1, int x, int y)
{
    float2 tileB = float2(x, y);
    float2 tileE = tileB + 1.0;

    // Construct an AABB of this tile.
    AABB aabbTile;
    aabbTile.min = float3(tileB * _TileSizeSS - 1.0, 0.0);
    aabbTile.max = float3(tileE * _TileSizeSS - 1.0, 1.0);

    // Get the tile's center.
    float3 center = aabbTile.Center();

    // Re-use the coverage computation to factor in strand width.
    float unused;
    float d = ComputeSegmentCoverageAndBarycentericCoordinate(center.xy, p0.xy, p1.xy, unused);

    // Compute the segment coverage provided by the segment distance.
    float coverage = 1 - smoothstep(0.0, 0.02, d);

    return any(coverage);
}

// Approach:
// 1) Build an axis-aligned bounding box per segments/thread.
// 2) Compute the tile coverage of the segment AABB.
// 3) Loop over each tile and determine the line coverage on the tile.
// 4) Build a linked-list per-tile of the resident segments.
[numthreads(16, 1, 1)]
void CoarsePass(uint3 i : SV_DispatchThreadID)
{
    if (i.x >= _SegmentCount)
        return;

    // Load Indices
    const uint i0 = _IndexBuffer[i.x * 2 + 0];
    const uint i1 = _IndexBuffer[i.x * 2 + 1];

    // Load Vertices
    const Vertex v0 = _VertexBuffer[i0];
    const Vertex v1 = _VertexBuffer[i1];

    // Invoke Vertex Shader
    float3 positionOS0, positionOS1;
    const float4 h0 = Vert(uint(v0.vertexID), positionOS0);
    const float4 h1 = Vert(uint(v1.vertexID), positionOS1);

    // Clip if behind the projection plane.
    if (h0.z > 0 || h1.z > 0)
        return;

    // Perspective divide
    float3 p0 = h0.xyz / h0.w;
    float3 p1 = h1.xyz / h1.w;

    // Compute the segment's AABB
    AABB aabb;
    aabb.min = min(p0, p1);
    aabb.max = max(p0, p1);

    // Transform AABB into tile space.
    int2 tilesBegin = ((aabb.min.xy * 0.5 + 0.5) * _ScreenParams.xy) / _TileSize;
    int2 tilesEnd   = ((aabb.max.xy * 0.5 + 0.5) * _ScreenParams.xy) / _TileSize;

    tilesBegin = clamp(tilesBegin, int2(0, 0), uint2(_GroupDim) - 1);
    tilesEnd   = clamp(tilesEnd,   int2(0, 0), uint2(_GroupDim) - 1);

    // Evaluate the coverage of each tile for this segment.
    for (int x = tilesBegin.x; x <= tilesEnd.x; ++x)
    {
        for (int y = tilesBegin.y; y <= tilesEnd.y; ++y)
        {
            if (!SegmentIntersectsTile(p0, p1, x, y))
                continue;

            uint tileID = y * _GroupDim.x + x;

            uint unused;
            InterlockedAdd(_SegmentCountBuffer[tileID], 1, unused);

            // Create a new segment entry node.
            SegmentData data;
            data.P0 = float4(p0, v0.vertexUV);
            data.P1 = float4(p1, v1.vertexUV);
            data.index = i.x * 2;

            // Retrieve global count segment count and iterate counter.
            // NOTE: UAVs don't have a counter, emulate the behavior with an extra counter resource.
#if 0
            uint segmentCount = _TileSegmentDataBuffer.IncrementCounter();
#else
            uint segmentCount;
            InterlockedAdd(_TileSegmentDataCounter[0], 1, segmentCount);
#endif

            // Exchange the new head pointer.
            uint next;
            _TileHeadPointerBuffer.InterlockedExchange(4 * tileID, segmentCount, next);

            // Add new segment entry to the tile segment linked list.
            TileListNode node;
            node.data  = data;
            node.next  = next;
            _TileSegmentDataBuffer[segmentCount] = node;
        }
    }
}

cbuffer ConstantsFinePass : register(b0)
{
    float4 _FinePassScreenParams;
    float4 GroupDim;
}

Buffer<uint>                   _FinePassHeadPointerBuffer : register(t0);
StructuredBuffer<TileListNode> _FinePassSegmentDataBuffer : register(t1);

#define FINE_RASTERIZER_MAX_DEPTH 64
#define FINE_RASTERIZER_LAST_NODE 0xFFFFFFFF

#undef _StrandCount
#undef _StrandParticleCount
#undef _PerStrandVertexCount
#undef _PerStrandSegmentCount
#undef _PerStrandIndexCount
#undef _IndexCount

#define _StrandCount           GroupDim.z
#define _StrandParticleCount   GroupDim.w
#define _PerStrandVertexCount  _StrandParticleCount
#define _PerStrandSegmentCount (_PerStrandVertexCount - 1)
#define _PerStrandIndexCount   (_PerStrandSegmentCount * 2)
#define _IndexCount            _StrandCount * _PerStrandIndexCount

// ported from GLSL to HLSL
// Also see:  https://www.shadertoy.com/view/4sfGzS
float iqhash( float n )
{
    return frac(sin(n)*43758.5453);
}

// Approach:
[numthreads(16, 16, 1)]
void FinePass(uint3 dispatchThreadID : SV_DispatchThreadID,
              uint3 groupID          : SV_GroupID)
{
    // Convert the dispatch coordinates to the generation space [0, 1]
    const float2 UV = ((float2)dispatchThreadID.xy + 0.5) * _FinePassScreenParams.zw;
    const float2 UVh = -1 + 2 * UV;

    // Compute the flattened group ID.
    uint tileID = (groupID.y * GroupDim.x) + groupID.x;

    // Grab the head pointer for the tile.
    uint next = _FinePassHeadPointerBuffer[tileID];

    float Z = -FLT_MAX;
    float4 result = 0;

    // Create and initialize the blending array.
    Fragment B[LAYERS_PER_PIXEL + 1];
    InitializeBlendingArray(B);

    while (next != FINE_RASTERIZER_LAST_NODE)
    {
         TileListNode node = _FinePassSegmentDataBuffer[next];

         // Compute the "barycenteric" coordinate on the segment.
         float coord;
         float d = ComputeSegmentCoverageAndBarycentericCoordinate(UVh, node.data.P0.xy, node.data.P1.xy, coord);

         // Compute the segment coverage provided by the segment distance.
         float coverage = 1 - smoothstep(0.0, 0.001, d);

         float2 coords = float2(
            coord,
            1 - coord
         );

// #if PERSPECTIVE_CORRECT_INTERPOLATION
//          // Ensure perspective correct coordinates.
//          const float2 w = rcp(float2(node.data.P0.w, node.data.P1.w));
//          coords = (coords * w) / dot(coords, w);
// #endif

         // Interpolate Vertex Data
         const float z = INTERP(coords, node.data.P0.z, node.data.P1.z);
         const float uv = INTERP(coords, node.data.P0.w, node.data.P1.w);

         if (coverage > 0 && z > Z)
         {
            float4 fragmentValue = float4(float3(1, 0, 0), 0.1);
            result = fragmentValue;
            Z = z;

            //float alpha = coverage * fragmentValue.a;
//
            //Fragment f;
            //f.a = fragmentValue.rgb * alpha;
            //f.t = 1.0 - alpha;
            //f.z = z;
//
            //InsertFragment(f, B);
         }

         next = node.next;
    }

    float transmittance = 1;

//   for (int k = 0; k < LAYERS_PER_PIXEL + 1; k++)
//   {
//       result.rgb = result.rgb + transmittance * B[k].a;
//       transmittance *= B[k].t;
//   }

    _OutputTarget[dispatchThreadID.xy] = result;
}