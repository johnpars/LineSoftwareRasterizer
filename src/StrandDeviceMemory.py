import math
import coalpy.gpu as gpu
import numpy as np

from src import Budgets
from src import Utility


class StrandDeviceMemory:

    def __init__(self):

        self.b_vertices = gpu.Buffer(
            name="GlobalVertexBuffer",
            type=gpu.BufferType.Structured,
            stride=Budgets.BYTE_SIZE_VERTEX_FORMAT,
            element_count=math.ceil(Budgets.BYTE_SIZE_VERTEX_POOL / Budgets.BYTE_SIZE_VERTEX_FORMAT)
        )

        self.b_indices = gpu.Buffer(
            name="GlobalIndexBuffer",
            type=gpu.BufferType.Standard,
            format=gpu.Format.R32_UINT,
            element_count=math.ceil(Budgets.BYTE_SIZE_INDEX_POOL / Budgets.BYTE_SIZE_INDEX_FORMAT)
        )

        self.b_strands = gpu.Buffer(
            name="GlobalStrandPositionBuffer",
            type=gpu.BufferType.Structured,
            stride=Budgets.BYTE_SIZE_STRAND_DATA_FORMAT,
            element_count=math.ceil(
                Budgets.BYTE_SIZE_STRAND_DATA_POOL / Budgets.BYTE_SIZE_STRAND_DATA_FORMAT)
        )

    def layout(self, strand_count, strand_particle_count):

        perLineVertices = strand_particle_count
        perLineSegments = perLineVertices - 1
        perLineIndices = perLineSegments * 2

        vertexID = np.zeros(strand_count * perLineVertices, dtype='f')
        vertexIDIdx = 0
        vertexUV = np.zeros(strand_count * perLineVertices, dtype='f')
        vertexUVIdx = 0
        indices = np.zeros(strand_count * perLineIndices, dtype='i')
        indicesIdx = 0

        unormU0 = int(65535 * 0.5)
        unormVk = int(65535 / perLineSegments)

        # Vertex ID
        i, k = 0, 0
        while i != strand_count:

            begin, stride, end = Utility.get_strand_iterator(Utility.MemoryLayout.Interleaved,
                                                             i, strand_count, strand_particle_count)

            j = begin
            while (j < end):
                vertexID[vertexIDIdx] = k

                k += 1
                vertexIDIdx += 1
                j += stride

            i += 1

        # Vertex UV
        i = 0
        while i != strand_count:

            begin, stride, end = Utility.get_strand_iterator(Utility.MemoryLayout.Interleaved,
                                                             i, strand_count, strand_particle_count)

            j, k = begin, 0
            while j < end:
                unormV = unormVk * k
                vertexUV[vertexUVIdx] = ((unormV << 16) | unormU0) / 0xffffffff  # Lazy integer normalization
                k += 1
                vertexUVIdx += 1
                j += stride

            i += 1

        # Interleave these IDs and UVs lists for buffer upload
        # Temporarily just write everything as floats and type cast in HLSL
        vertices = np.array([val for pair in zip(vertexID, vertexUV) for val in pair], dtype='f')

        # Indices
        i, s = 0, 0
        while i != strand_count:
            j = 0
            while j != perLineSegments:
                indices[indicesIdx] = s + 0
                indicesIdx += 1
                indices[indicesIdx] = s + 1
                indicesIdx += 1
                j += 1
                s += 1
            i += 1
            s += 1

        # Upload

        cmd = gpu.CommandList()

        cmd.upload_resource(
            source=vertices,
            destination=self.b_vertices
        )

        cmd.upload_resource(
            source=indices,
            destination=self.b_indices
        )

        gpu.schedule(cmd)

        return

    def bind_strand_position_data(self, positions: np.ndarray):

        # Dumb flattening of the object list
        positionsGPU = np.zeros(positions.size * 3, 'f')

        i, j = 0, 0
        while i != positions.size * 3:
            positionsGPU[i + 0] = positions[j].x
            positionsGPU[i + 1] = positions[j].y
            positionsGPU[i + 2] = positions[j].z
            i += 3
            j += 1

        cmd = gpu.CommandList()

        cmd.upload_resource(
            source=positionsGPU,
            destination=self.b_strands
        )

        gpu.schedule(cmd)

        return
