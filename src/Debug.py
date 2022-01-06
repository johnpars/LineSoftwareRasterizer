import math
import coalpy.gpu as gpu
import numpy as np

from dataclasses import dataclass
from src import StrandRasterizer

TextureFont = gpu.Texture(file="DebugFont.jpg")
SamplerFont = gpu.Sampler(filter_type=gpu.FilterType.Linear)

ShaderDebugCountSegmentSetup = gpu.Shader(file="Debug.hlsl", name="CountSegmentSetup", main_function="CountSegmentSetup")
ShaderDebugSegmentsPerTile = gpu.Shader(file="Debug.hlsl", name="SegmentsPerTile", main_function="SegmentsPerTile")


@dataclass
class Stats:
    segmentCount: int
    segmentCountPassedFrustumCull: int


def ComputeStats(rasterizer, context) -> Stats:
    cmd = gpu.CommandList()

    output = gpu.Buffer(
        type=gpu.BufferType.Standard,
        format=gpu.Format.R32_UINT,
        element_count=1
    )

    cmd.dispatch(
        x=math.ceil(context.segmentCount / 64),
        inputs=[
            rasterizer.mSegmentCountBuffer
        ],
        outputs=output,
        shader=ShaderDebugCountSegmentSetup
    )

    gpu.schedule(cmd)

    # Read back and report the result.
    download = gpu.ResourceDownloadRequest(output)
    download.resolve()
    result = np.frombuffer(download.data_as_bytearray(), dtype='i')

    return Stats(
        context.segmentCount, result[0]
    )


def SegmentsPerTile(cmd, outputTarget, w, h, rasterizer: StrandRasterizer):
    cmd.begin_marker("DebugSegmentsPerTile")

    groupDimX = math.ceil(w / rasterizer.CoarseTileSize)
    groupDimY = math.ceil(h / rasterizer.CoarseTileSize)

    cmd.dispatch(
        shader=ShaderDebugSegmentsPerTile,

        constants=[
            groupDimX,
            groupDimY
        ],

        inputs=[
            TextureFont,
            rasterizer.mCoarseTileSegmentCount
        ],

        outputs=outputTarget,

        samplers=SamplerFont,

        x=math.ceil(w / 16),
        y=math.ceil(h / 16),
        z=1
    )

    cmd.end_marker()
