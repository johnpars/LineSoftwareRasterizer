import math
import numpy as np
import coalpy.gpu as gpu

from . import StrandDeviceMemory

CoreShader = gpu.Shader(file = "shaders/StrandRasterizer.hlsl", name = "StrandRasterizer", main_function = "Main")

def Go(cmd, strands : StrandDeviceMemory, target : gpu.Texture, matrixV, matrixP, w, h):

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
            [ 0.0, 0.0, 0.0, 0.0 ]
        ], dtype='f'),

        x = math.ceil(w / 8),
        y = math.ceil(h / 8),

        inputs = [
            strands.mVertexBuffer,
            strands.mIndexBuffer
        ],

        outputs = target
    )