import math
import numpy as np
import coalpy.gpu as gpu

from src import StrandDeviceMemory

CoreShader = gpu.Shader(file = "StrandRasterizer.hlsl", name = "StrandRasterizer", main_function = "Main")

def Go(cmd, strandCount, strandParticleCount, strands : StrandDeviceMemory, target : gpu.Texture, matrixV, matrixP, w, h):

    cmd.dispatch(
        shader = CoreShader,

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
            [ strandCount, strandParticleCount, strandCount * (strandParticleCount - 1), 0.0 ]
        ], dtype='f'),

        x = math.ceil(w / 8),
        y = math.ceil(h / 8),

        inputs = [
            strands.mVertexBuffer,
            strands.mIndexBuffer,
            strands.mPositionBuffer
        ],

        outputs = target
    )