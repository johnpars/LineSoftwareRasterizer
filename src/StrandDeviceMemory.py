import array
import ctypes.wintypes
import math
import sys

import coalpy.gpu as gpu
import numpy as np

from src import Utility

class StrandDeviceMemory:

    # VBO
    kVertexPoolByteSize   = 512 * 1024 * 1024                # 32mb
    kVertexFormatByteSize = 4 + 4                           # Vertex ID + Vertex UV

    # IBO
    kIndexPoolByteSize    = 512 * 1024 * 1024                # 16mb
    kIndexFormatByteSize  = 4                               # 32 bit

    # Strand Positions
    kStrandPositionPoolByteSize   = 512 * 1024 * 1024        # 32mb
    kStrandPositionFormatByteSize = 4 * 3                   # Position


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

        self.mStrandDataBuffer = gpu.Buffer(
            name = "GlobalStrandPositionBuffer",
            type = gpu.BufferType.Structured,
            stride = StrandDeviceMemory.kStrandPositionFormatByteSize,
            element_count = math.ceil(StrandDeviceMemory.kStrandPositionPoolByteSize / StrandDeviceMemory.kStrandPositionFormatByteSize)
        )


    def Layout(self, strandCount, strandParticleCount):

        perLineVertices = strandParticleCount
        perLineSegments = perLineVertices - 1
        perLineIndices  = perLineSegments * 2

        vertexID = np.zeros(strandCount * perLineVertices, dtype='f'); vertexIDIdx = 0
        vertexUV = np.zeros(strandCount * perLineVertices, dtype='f'); vertexUVIdx = 0
        indices  = np.zeros(strandCount * perLineIndices,  dtype='i'); indicesIdx  = 0

        unormU0 = int(65535 * 0.5)
        unormVk = int(65535 / perLineSegments)

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

        # Vertex UV
        i = 0
        while i != strandCount:

            begin, stride, end = Utility.GetStrandIterator(Utility.MemoryLayout.Interleaved,
                                                           i, strandCount, strandParticleCount)

            j, k = begin, 0
            while (j < end):
                unormV = unormVk * k
                vertexUV[vertexUVIdx] = ((unormV << 16) | unormU0) / 0xffffffff # Lazy integer normalization
                k += 1; vertexUVIdx += 1; j += stride

            i += 1

        # Interleave these IDs and UVs lists for buffer upload
        # Temporarily just write everything as floats and type cast in HLSL
        vertices = np.array([val for pair in zip(vertexID, vertexUV) for val in pair], dtype='f')

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
            source = vertices,
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
            destination = self.mStrandDataBuffer
        )

        gpu.schedule(cmd)

        return