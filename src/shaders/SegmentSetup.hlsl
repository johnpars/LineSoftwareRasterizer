#include "RasterCommon.hlsl"
#include "Vert.hlsl"

// Inputs
// ----------------------------------------
cbuffer ConstantsSetup : register(b0)
{
    float4 _Params;
}

// TODO: ByteAddress for faster load and store in SGPR
StructuredBuffer<VertexInput> _VertexBuffer     : register(t0);
Buffer<uint>                  _IndexBuffer      : register(t1);

// Outputs
// ----------------------------------------
RWStructuredBuffer<uint>          _SegmentCountBuffer  : register(u0);
RWStructuredBuffer<SegmentRecord> _SegmentRecordBuffer : register(u1);

// Defines
// ----------------------------------------
#define _SegmentCount _Params.x

// Defines
// ----------------------------------------

#define INSIDE 0 // 0000
#define LEFT   1 // 0001
#define RIGHT  2 // 0010
#define BOTTOM 4 // 0100
#define TOP    8 // 1000

// TODO: Currently we perform the clipping in NDC, figure out how to do it in homogenous coordinates instead.
#define MIN_X -1
#define MAX_X +1
#define MIN_Y -1
#define MAX_Y +1

#define NUM_WARP 8

// Utility
//-----------------------------------------

uint ComputeOutCode(float x, float y)
{
    uint code = INSIDE;
    {
        if      (x < MIN_X) { code |= LEFT;   }
        else if (x > MAX_X) { code |= RIGHT;  }
        if      (y < MIN_Y) { code |= BOTTOM; }
        else if (y > MAX_Y) { code |= TOP;    }
    }
	return code;
}

// TODO: Investigate "Improvement in the Cohen-Sutherland Line Segment Clipping Algorithm" for something faster.
bool ClipSegmentCohenSutherland(inout float x0, inout float y0, inout float x1, inout float y1)
{
    uint outCode0 = ComputeOutCode(x0, y0);
    uint outCode1 = ComputeOutCode(x1, y1);

    bool accept = false;

    for(;;)
    {
        // Trivially accept, both points inside the window.
        if(!(outCode0 | outCode1))
        {
            accept = true;
            break;
        }
        // Trivially reject, both points outside the window.
        else if(outCode0 & outCode1)
        {
            break;
        }
        // Both tests failed, calculate the clipped segment.
        else
        {
            // One point is outside the window. Need to compute a new point clipped to the window edge.
            float x, y;

            // Choose the out code that is outside the window.
            uint outCodeOut = outCode1 > outCode0 ? outCode1 : outCode0;

            // Determine the clipped position based on the out code.
            if      (outCodeOut & TOP)    { x = x0 + (x1 - x0) * (MAX_Y - y0) / (y1  - y0); y = MAX_Y; }
            else if (outCodeOut & BOTTOM) { x = x0 + (x1 - x0) * (MIN_Y - y0) / (y1  - y0); y = MIN_Y; }
            else if (outCodeOut & RIGHT)  { y = y0 + (y1 - y0) * (MAX_X - x0) / (x1  - x0); x = MAX_X; }
            else if (outCodeOut & LEFT)   { y = y0 + (y1 - y0) * (MIN_X - x0) / (x1  - x0); x = MIN_X; }

            if (outCodeOut == outCode0)
            {
                x0 = x;
                y0 = y;
                outCode0 = ComputeOutCode(x0, y0);
            }
            else
            {
                x1 = x;
                y1 = y;
                outCode1 = ComputeOutCode(x1, y1);
            }
        }
    }

    return accept;
}

// Kernel:
// For every segment, clip, cull, tessellate, compute data.
// The mapping is 1-1, so we tag each segment with the amount of processing required.
// (0 for culled/clipped, 1 for a single segment, >1 for subsegments if tessellated).
// ----------------------------------------
[numthreads(NUM_WARP * NUM_THREAD_PER_WARP, 1, 1)]
void SegmentSetup(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint i = dispatchThreadID.x;

    if (i >= _SegmentCount)
        return;

    // Prepare the segment record.
    SegmentRecord record;
    ZERO_INITIALIZE(SegmentRecord, record);

    // Load Indices
    const uint i0 = _IndexBuffer[i * 2 + 0];
    const uint i1 = _IndexBuffer[i * 2 + 1];

    // Load Vertices
    const VertexInput i_v0 = _VertexBuffer[i0];
    const VertexInput i_v1 = _VertexBuffer[i1];

    // Invoke Vertex Shader
    // TODO: Add a preliminary vertex stage that processes vertex attributes and writes to a vertex buffer.
    //       Store vertex indices in the segment data which we can read back later in the fine rasterizer for
    //       arbitrary vertex data interpolation.
    VertexOutput o_v0 = Vert(i_v0);
    VertexOutput o_v1 = Vert(i_v1);

    float4 v[2] = {
        o_v0.positionCS,
        o_v1.positionCS
    };

    // Fast rejection for segments behind the near clipping plane.
    if (0 < v[0].w || 0 < v[1].w)
    {
        _SegmentCountBuffer[i]  = 0;
        _SegmentRecordBuffer[i] = record;
        return;
    }

    // Perspective divide. Homogenous -> NDC.
    float3 p0 = v[0].xyz / v[0].w;
    float3 p1 = v[1].xyz / v[1].w;

    // Cohen-Sutherland algorithm to perform line segment clipping in NDC space.
    if(!ClipSegmentCohenSutherland(p0.x, p0.y, p1.x, p1.y))
    {
        _SegmentCountBuffer[i]  = 0;
        _SegmentRecordBuffer[i] = record;
        return;
    }

    // NOTE: This can potentially expand to greater than one if we tessellate the segment.
    _SegmentCountBuffer[i] = 1;

    // Record the segment.
    {
        record.v0 = p0.xy;
        record.v1 = p1.xy;
    }
    _SegmentRecordBuffer[i] = record;
}