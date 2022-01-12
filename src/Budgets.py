# GPU Memory Allocation Budgets, and Hardware Resource Limits.

# Geometry + Input
# --------------------------------------------------------------

# Primitive Cap
MAX_SEGMENTS                    = 1 << 22

# VBO
BYTE_SIZE_VERTEX_POOL           = 32 * 1024 * 1024
BYTE_SIZE_VERTEX_FORMAT         = 4 + 4

# IBO
BYTE_SIZE_INDEX_POOL            = 32 * 1024 * 1024
BYTE_SIZE_INDEX_FORMAT          = 4

# Strand Data
BYTE_SIZE_STRAND_DATA_POOL      = 32 * 1024 * 1024
BYTE_SIZE_STRAND_DATA_FORMAT    = 4 * 3

# Geometry Processing
# --------------------------------------------------------------

# Vertex Output
BYTE_SIZE_VERTEX_OUTPUT_POOL    = 16 * 1024 * 1024
BYTE_SIZE_VERTEX_OUTPUT_FORMAT  = 4 * 4

# Segment Setup
BYTE_SIZE_SEGMENT_POOL          = 32 * 1024 * 1024
BYTE_SIZE_SEGMENT_HEADER_FORMAT = 4 * 4
BYTE_SIZE_SEGMENT_DATA_FORMAT   = 4 * 2

# Binning
# --------------------------------------------------------------

BYTE_SIZE_BIN_RECORD_POOL       = 128 * 1024 * 1024
BYTE_SIZE_BIN_RECORD_FORMAT     = 4 + 4 + 4
TILE_SIZE_BIN                   = 16

# Work Queue
# --------------------------------------------------------------

BYTE_SIZE_WORK_QUEUE_POOL       = 128 * 1024 * 1024
BYTE_SIZE_WORK_QUEUE_FORMAT     = 4

# Brute
# --------------------------------------------------------------

# Color, coverage, depth, next
BYTE_SIZE_FRAGMENT_DATA_FORMAT  = (4 * 4) + 4 + 4

# Allow roughly 4 fragment list depth for every pixel in a 1920x1080 resolution window.
BYTE_SIZE_FRAGMENT_DATA_POOL    = 200 * 1024 * 1024

# Hardware
# --------------------------------------------------------------

NUM_CU              = 16
NUM_WAVE_PER_CU     = 40
NUM_LANE_PER_WAVE   = 64
