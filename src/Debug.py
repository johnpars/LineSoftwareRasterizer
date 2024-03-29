import math
import coalpy.gpu as gpu
import numpy as np

from dataclasses import dataclass
from src import Rasterizer
from src import Utility
from src import Budgets

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
            self.b_frustum_segment_output,
            Utility.ClearMode.UINT
        )

        cmd.dispatch(
            x=math.ceil(context.segment_count / Budgets.NUM_LANE_PER_WAVE),
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
    def draw_bin_counts(cmd, rasterizer, context, opacity):
        cmd.begin_marker("DebugSegmentsPerTile")

        group_dim_y = math.ceil(context.h / Budgets.TILE_SIZE_BIN)
        group_dim_x = math.ceil(context.w / Budgets.TILE_SIZE_BIN)

        cmd.dispatch(
            shader=s_segments_per_tile,

            constants=[
                group_dim_x,
                group_dim_y,
                opacity
            ],

            inputs=[
                TextureFont,
                rasterizer.b_bin_counters
            ],

            outputs=context.target,
            samplers=SamplerFont,

            x=math.ceil(context.w / Budgets.TILE_SIZE_BIN),
            y=math.ceil(context.h / Budgets.TILE_SIZE_BIN),
            z=1
        )

        cmd.end_marker()
