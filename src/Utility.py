import math

import coalpy.gpu as gpu

ShaderClearTarget = gpu.Shader(file = "shaders/ClearTarget.hlsl", name = "ClearTarget", main_function = "Main")

def ClearTarget(cmd, color, target, w, h):
    cmd.dispatch(
        shader = ShaderClearTarget,
        constants = color,
        x = math.ceil(w / 8),
        y = math.ceil(h / 8),
        z = 1,
        outputs = target
    )