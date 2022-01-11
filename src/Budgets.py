# GPU Memory Allocation Budgets, and Hardware Limits.

# VBO
BYTE_SIZE_VERTEX_POOL           = 32 * 1024 * 1024
BYTE_SIZE_VERTEX_FORMAT         = 4 + 4

# IBO
BYTE_SIZE_INDEX_POOL            = 32 * 1024 * 1024
BYTE_SIZE_INDEX_FORMAT          = 4

# Strand Data
BYTE_SIZE_STRAND_DATA_POOL      = 32 * 1024 * 1024
BYTE_SIZE_STRAND_DATA_FORMAT    = 4 * 3

# Vertex Output
BYTE_SIZE_VERTEX_OUTPUT_POOL    = 16 * 1024 * 1024
BYTE_SIZE_VERTEX_OUTPUT_FORMAT  = 4 * 4

# Segment Setup
BYTE_SIZE_SEGMENT_POOL          = 32 * 1024 * 1024
BYTE_SIZE_SEGMENT_HEADER_FORMAT = (4 * 4)
BYTE_SIZE_SEGMENT_DATA_FORMAT   = (4 * 2)

# Tile Segment Buffer
MAX_SEGMENTS                    = 1 << 22
TILE_SIZE_COARSE                = 16

# Color, coverage, depth, next
BYTE_SIZE_FRAGMENT_DATA_FORMAT  = (4 * 4) + 4 + 4

# Allow roughly 4 fragment list depth for every pixel in a 1920x1080 resolution window.
BYTE_SIZE_FRAGMENT_DATA_POOL    = 200 * 1024 * 1024

# Counters
NUM_COUNTERS = 2

# Hardware Specific Budgets (AMD Radeon Pro 460)
NUM_CU              = 16
NUM_WAVE_PER_CU     = 40
NUM_LANE_PER_WAVE   = 64