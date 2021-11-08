import math

import coalpy.gpu as gpu

from . import StrandGroupGPU

CoreShader = gpu.Shader(file = "shaders/StrandRasterizer.hlsl", name = "StrandRasterizer", main_function = "Main")

def Go( cmd, strands : StrandGroupGPU, target : gpu.Texture, w, h):

    cmd.dispatch(
        shader = CoreShader,
        constants = [

        ],
        x = math.ceil(w / 8),
        y = math.ceil(h / 8),
        inputs = [
            strands.mVertexBuffer,
            strands.mIndexBuffer
        ],
        outputs = target
    )