import array
import math

import coalpy.gpu as gpu
import numpy as np

from src import Utility

class StrandDeviceMemory:

    # VBO
    kVertexPoolByteSize   = 32 * 1024 * 1024                # 32mb
    kVertexFormatByteSize = 4 * 3                           # Position

    # IBO
    kIndexPoolByteSize    = 16 * 1024 * 1024                # 16mb
    kIndexFormatByteSize  = 4                               # 32 bit

    # Strand Positions
    kStrandPositionPoolByteSize   = 32 * 1024 * 1024                # 32mb
    kStrandPositionFormatByteSize = 4 * 3                           # Position


    def __init__(self):

        self.mVertexBuffer = gpu.Buffer(
            name = "GlobalVertexBuffer",
            type = gpu.BufferType.Standard,
            format = gpu.Format.R32_UINT,
            element_count = math.ceil(StrandDeviceMemory.kVertexPoolByteSize / StrandDeviceMemory.kVertexFormatByteSize)
        )

        self.mIndexBuffer = gpu.Buffer(
            name = "GlobalIndexBuffer",
            type = gpu.BufferType.Standard,
            format = gpu.Format.R32_UINT,
            element_count = math.ceil(StrandDeviceMemory.kIndexPoolByteSize / StrandDeviceMemory.kIndexFormatByteSize)
        )

        self.mPositionBuffer = gpu.Buffer(
            name = "GlobalStrandPositionBuffer",
            type = gpu.BufferType.Structured,
            stride = StrandDeviceMemory.kStrandPositionFormatByteSize,
            element_count = math.ceil(StrandDeviceMemory.kStrandPositionPoolByteSize / StrandDeviceMemory.kStrandPositionFormatByteSize)
        )


    def Layout(self, strandCount, strandParticleCount):

        perLineVertices = strandParticleCount
        perLineSegments = perLineVertices - 1
        perLineIndices  = perLineSegments * 2

        vertexID = np.zeros(strandCount * perLineVertices, dtype='i'); vertexIDIdx = 0
        # vertexUV = np.empty(strandCount * perLineVertices, dtype='f'); vertexUVIdx = 0
        indices  = np.zeros(strandCount * perLineIndices,  dtype='i'); indicesIdx  = 0

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
                indices[indicesIdx] = s + 0; indicesIdx += 1
                indices[indicesIdx] = s + 1; indicesIdx += 1
                j += 1; s += 1
            i += 1; s += 1

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

    def BindStrandPositionData(self, positions : np.ndarray):

        # Dumb flattening of the object list
        positionsGPU = np.zeros(positions.size * 3, 'f')

        i, j = 0, 0
        while i != positions.size * 3:
            positionsGPU[i + 0] = positions[j].x
            positionsGPU[i + 1] = positions[j].y
            positionsGPU[i + 2] = positions[j].z
            i += 3; j += 1

        cmd = gpu.CommandList()

        cmd.upload_resource(
            source = positionsGPU,
            destination = self.mPositionBuffer
        )

        gpu.schedule(cmd)

        return