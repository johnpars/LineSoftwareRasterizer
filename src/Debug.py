import math
import coalpy.gpu as gpu

from src import StrandRasterizer

TextureFont = gpu.Texture(file = "DebugFont.jpg")
SamplerFont = gpu.Sampler(filter_type = gpu.FilterType.Linear)

ShaderDebugSegmentsPerTile = gpu.Shader(file = "Debug.hlsl", name = "SegmentsPerTile", main_function ="SegmentsPerTile")


def SegmentsPerTile(cmd, outputTarget, w, h, rasterizer : StrandRasterizer):
    cmd.begin_marker("DebugSegmentsPerTile")

    groupDimX = math.ceil(w / rasterizer.CoarseTileSize)
    groupDimY = math.ceil(h / rasterizer.CoarseTileSize)

    cmd.dispatch(
        shader= ShaderDebugSegmentsPerTile,

        constants = [
            groupDimX,
            groupDimY
        ],

        inputs = [
            TextureFont,
            rasterizer.mCoarseTileSegmentCount
        ],

        outputs = outputTarget,

        samplers = SamplerFont,

        x = math.ceil(w / 16),
        y = math.ceil(h / 16),
        z = 1
    )

    cmd.end_marker()