import math
import coalpy.gpu as gpu
import numpy as np

from dataclasses import dataclass
from src import Rasterizer
from src import Utility

TextureFont = gpu.Texture(file="DebugFont.jpg")
SamplerFont = gpu.Sampler(filter_type=gpu.FilterType.Linear)

s_count_clipped_segments = gpu.Shader(file="debug/DebugCountClippedSegments.hlsl", name="CountClippedSegments", main_function="CountClippedSegments")
s_segments_per_tile      = gpu.Shader(file="debug/DebugSegmentsPerTile.hlsl", name="SegmentsPerTile", main_function="SegmentsPerTile")


@dataclass
class Stats:
    segmentCount: int
    segmentCountPassedFrustumCull: int


class Debug:
    def __init__(self):
        self.b_frustum_segment_output = gpu.Buffer(
            type=gpu.BufferType.Standard,
            format=gpu.Format.R32_UINT,
            element_count=1
        )

    def compute_stats(self, cmd, rasterizer, context) -> Stats:
        Utility.clear_buffer(
            cmd,
            0,
            1,
            self.b_frustum_segment_output
        )

        cmd.dispatch(
            x=math.ceil(context.segment_count / 64),
            inputs=[
                rasterizer.b_segment_output
            ],
            outputs=self.b_frustum_segment_output,
            shader=s_count_clipped_segments
        )

        # Read back and report the result.
        download = gpu.ResourceDownloadRequest(self.b_frustum_segment_output)
        download.resolve()
        result = np.frombuffer(download.data_as_bytearray(), dtype='i')

        return Stats(
            context.segment_count, result[0]
        )

    @staticmethod
    def segments_per_tile(cmd, output_target, w, h, rasterizer: Rasterizer):
        cmd.begin_marker("DebugSegmentsPerTile")

        group_dim_x = math.ceil(w / rasterizer.TILE_SIZE_COARSE)
        group_dim_y = math.ceil(h / rasterizer.TILE_SIZE_COARSE)

        cmd.dispatch(
            shader=s_segments_per_tile,

            constants=[
                group_dim_x,
                group_dim_y
            ],

            inputs=[
                TextureFont,
                rasterizer.b_coarse_tile_count
            ],

            outputs=output_target,

            samplers=SamplerFont,

            x=math.ceil(w / 16),
            y=math.ceil(h / 16),
            z=1
        )

        cmd.end_marker()
