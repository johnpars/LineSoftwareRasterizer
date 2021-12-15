// Defines
// -----------------------------------------------------

// Hardware specific metrics, based on Nvidia Quadro RTX 8000.
#define NUM_SM              72
#define NUM_WARP            32
#define NUM_THREAD_PER_WARP 32

#define GROUP_SIZE_1D       NUM_WARP * NUM_THREAD_PER_WARP

#define ZERO_INITIALIZE(type, name) name = (type)0;

// Structures
// -----------------------------------------------------
struct StrandData
{
    float3 strandPositionOS;
};

struct VertexInput
{
    float vertexID;
    float vertexUV;
};

struct VertexOutput
{
    float4 positionCS;
};

struct SegmentData
{
    VertexOutput v0;
    VertexOutput v1;
};