import math
import coalpy.gpu as gpu

import StrandRasterizer

TextureFont = gpu.Texture(file = "DebugFont.jpg")
SamplerFont = gpu.Sampler(filter_type = gpu.FilterType.Linear)

ShaderDebugSegmentsPerTile = gpu.Shader(file = "Debug.hlsl", name = "SegmentsPerTile", main_function ="SegmentsPerTile")


def SegmentsPerTile(cmd, outputTarget, w, h, rasterizer : StrandRasterizer):
    cmd.begin_marker("DebugSegmentsPerTile")

    cmd.dispatch(
        shader= ShaderDebugSegmentsPerTile,

        inputs = [
            TextureFont
        ],

        outputs = outputTarget,

        samplers = SamplerFont,

        x = math.ceil(w / 16),
        y = math.ceil(h / 16),
        z = 1
    )

    cmd.end_marker()