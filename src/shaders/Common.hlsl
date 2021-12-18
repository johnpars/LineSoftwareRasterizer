// Defines
// -----------------------------------------------------

// Hardware specific metrics, based on Nvidia Quadro RTX 8000.
#define NUM_SM              72
#define NUM_WARP_PER_SM     32
#define NUM_THREAD_PER_WARP 32

#define ZERO_INITIALIZE(type, name) name = (type)0;

// Counter indices
#define ATOMIC_COUNTER_BIN 0

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

struct SegmentHeader
{
    int2 v0;
    int2 v1;
};

struct SegmentData
{


    // Vertex Indices
    uint vi0;
    uint vi1;
};