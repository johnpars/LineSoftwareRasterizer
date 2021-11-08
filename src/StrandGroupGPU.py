import array
import math

import coalpy.gpu as gpu

class StrandGroupGPU:

    kVertexPoolByteSize   = 32 * 1024 * 1024                # 32mb
    kIndexPoolByteSize    = 16 * 1024 * 1024                # 16mb
    kVertexFormatByteSize = ((4 * 3) + (4 * 3) +  (4 * 2))  # Position, Normal, UV
    kIndexFormatByteSize  = 4                               # 32 bit

    def __init__(self):

        self.mVertexBuffer = gpu.Buffer(
            name = "GlobalVertexBuffer",
            type = gpu.BufferType.Structured,
            stride = StrandGroupGPU.kVertexFormatByteSize,
            element_count = math.ceil(StrandGroupGPU.kVertexPoolByteSize / StrandGroupGPU.kVertexFormatByteSize)
        )

        self.mIndexBuffer = gpu.Buffer(
            name = "GlobalIndexBuffer",
            type = gpu.BufferType.Standard,
            format = gpu.Format.R32_UINT,
            element_count = math.ceil(StrandGroupGPU.kIndexPoolByteSize / StrandGroupGPU.kIndexFormatByteSize)
        )

    # Create a simple zig-zag in the X-Y plane. \
    # TODO: Let StrandGroup define the data and let StrandGroupGPU define the connectivity and allocations
    # (This will conform to the data model proposed by demo team)

    def CreateSimpleStrand(self):

        strandData = array.array('f', [
           # v.x,  v.y,  v.z
            -4.0,  0.0,  0.0,
            -2.0, +2.0,  0.0,
             0.0,  0.0,  0.0,
             2.0, -2.0,  0.0,
             4.0,  0.0,  0.0
        ])

        indexData = [ 0, 1, 2, 3, 4 ]

        cmd = gpu.CommandList()

        cmd.upload_resource(
            source = strandData,
            destination = self.mVertexBuffer
        )

        cmd.upload_resource(
            source = indexData,
            destination = self.mIndexBuffer
        )

        gpu.schedule([cmd])