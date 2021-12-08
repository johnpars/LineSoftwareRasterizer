import math
import numpy as np
import coalpy.gpu as gpu

from src import Utility
from src import StrandDeviceMemory

ShaderBruteForce = gpu.Shader(file ="StrandRasterizer.hlsl", name ="BruteForce", main_function ="BruteForce")
ShaderCoarsePass = gpu.Shader(file ="StrandRasterizer.hlsl", name ="CoarsePass", main_function ="CoarsePass")

class StrandRasterizer:

    CoarseTileSize = 16

    def __init__(self, w, h):

        self.mW = 0
        self.mH = 0
        self.UpdateResolutionDependentBuffers(w, h)

        return

    def UpdateResolutionDependentBuffers(self, w, h):

        if (w <= self.mW and h <= self.mH):
            return

        self.mW = w
        self.mH = h

        cW = math.ceil(w / StrandRasterizer.CoarseTileSize)
        cH = math.ceil(h / StrandRasterizer.CoarseTileSize)

        self.mCoarseTileSegmentCount = gpu.Buffer(
            name = "CoarseTileSegmentCount",
            type = gpu.BufferType.Standard,
            format = gpu.Format.R32_UINT,
            element_count = cW * cH
        )

    def BruteForce(self, cmd, strandCount, strandParticleCount, strands : StrandDeviceMemory, target : gpu.Texture, matrixV, matrixP, w, h):

        cmd.dispatch(
            shader = ShaderBruteForce,

            constants = np.array([
                # _MatrixV
                matrixV[0, 0:4],
                matrixV[1, 0:4],
                matrixV[2, 0:4],
                matrixV[3, 0:4],

                # _MatrixP
                matrixP[0, 0:4],
                matrixP[1, 0:4],
                matrixP[2, 0:4],
                matrixP[3, 0:4],

                # _ScreenParams
                [ w, h, 1.0 / w, 1.0 / h ],

                # _Params0
                [ strandCount, strandParticleCount, 0.0, 0.0 ]
            ], dtype='f'),

            x = math.ceil(w / 8),
            y = math.ceil(h / 8),

            inputs = [
                strands.mVertexBuffer,
                strands.mIndexBuffer,
                strands.mStrandDataBuffer
            ],

            outputs = target
        )

    def CoarsePass(self, cmd, strandCount, strands : StrandDeviceMemory, w, h):

        Utility.ClearBuffer(
            cmd,
            0,
            math.ceil(w / StrandRasterizer.CoarseTileSize) *
            math.ceil(h / StrandRasterizer.CoarseTileSize),
            self.mCoarseTileSegmentCount
        )

        cmd.dispatch(
            shader = ShaderCoarsePass,

            outputs = [
                self.mCoarseTileSegmentCount
            ],

            x = math.ceil(strandCount / StrandRasterizer.CoarseTileSize),
            y = 1,
            z = 1
        )

        return