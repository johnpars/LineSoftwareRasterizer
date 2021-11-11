import array
import math

import coalpy.gpu as gpu
import numpy as np

from src import Utility

class StrandDeviceMemory:

    kVertexPoolByteSize   = 12 * 1024 * 1024                # 32mb
    kIndexPoolByteSize    = 16 * 1024 * 1024                # 16mb
    kVertexFormatByteSize = 4 * 3                           # Position
    kIndexFormatByteSize  = 4                               # 32 bit

    def __init__(self):

        self.mVertexBuffer = gpu.Buffer(
            name = "GlobalVertexBuffer",
            type = gpu.BufferType.Structured,
            stride = StrandDeviceMemory.kVertexFormatByteSize,
            element_count = math.ceil(StrandDeviceMemory.kVertexPoolByteSize / StrandDeviceMemory.kVertexFormatByteSize)
        )

        self.mIndexBuffer = gpu.Buffer(
            name = "GlobalIndexBuffer",
            type = gpu.BufferType.Standard,
            format = gpu.Format.R32_UINT,
            element_count = math.ceil(StrandDeviceMemory.kIndexPoolByteSize / StrandDeviceMemory.kIndexFormatByteSize)
        )


    def Layout(self, strandCount, strandParticleCount):

        perLineVertices = strandParticleCount
        perLineSegments = perLineVertices - 1
        perLineIndices  = perLineSegments * 2

        vertexID = np.empty(strandCount * perLineVertices, dtype='i'); vertexIDIdx = 0
        # vertexUV = np.empty(strandCount * perLineVertices, dtype='f'); vertexUVIdx = 0
        indices  = np.empty(strandCount * perLineIndices,  dtype='i'); indicesIdx  = 0

        # Vertex ID
        i, k = 0, 0
        while i != strandCount:

            begin, stride, end = Utility.GetStrandIterator(Utility.MemoryLayout.Interleaved,
                                                           i, strandCount, strandParticleCount)

            j = begin
            while (j < end):
                vertexID[vertexIDIdx] = k

                k += 1; vertexIDIdx += 1; j += stride

            i += 1

        # Indices
        i, s = 0, 0
        while i != strandCount:
            j = 0
            while j != perLineSegments:
                indices[i + 0] = s + 0
                indices[i + 1] = s + 1
                j += 1
            i += 1
            s += 1

        # Upload

        cmd = gpu.CommandList()

        cmd.upload_resource(
            source = vertexID,
            destination = self.mVertexBuffer
        )

        cmd.upload_resource(
            source = indices,
            destination = self.mIndexBuffer
        )

        gpu.schedule(cmd)

        return
