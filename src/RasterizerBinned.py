import math
import numpy as np
import coalpy.gpu as gpu

from src import Utility
from src import Budgets
from src import Rasterizer

s_raster_bin    = gpu.Shader(file="RasterBin.hlsl",    name="RasterBin",    main_function="RasterBin")
s_raster_coarse = gpu.Shader(file="RasterCoarse.hlsl", name="RasterCoarse", main_function="RasterCoarse")
s_raster_fine   = gpu.Shader(file="RasterFine.hlsl",   name="RasterFine",   main_function="RasterFine")


class RasterizerBinned(Rasterizer.Rasterizer):
    def __init__(self, w, h):

        # Resources
        self.b_counters = None
        self.b_bin_count = None

        # Constant buffers
        self.cb_raster_bin = None

        super().__init__(w, h)

    def create_resource_buffers(self):
        super().create_resource_buffers()

        self.b_counters = gpu.Buffer(
            name="CounterBuffer",
            type=gpu.BufferType.Standard,
            format=gpu.Format.R32_UINT,
            element_count=1
        )

    def create_constant_buffers(self):
        super().create_constant_buffers()

        self.cb_raster_bin = gpu.Buffer(
            name="ConstantBufferRasterBin",
            type=gpu.BufferType.Structured,
            stride=(4 * 4),
            element_count=1,
            is_constant_buffer=True
        )

    def update_resolution_dependent_buffers(self, w, h):
        if w <= self.mW and h <= self.mH:
            return

        super().update_resolution_dependent_buffers(w, h)

        self.b_bin_count = gpu.Buffer(
            name="BinCountBuffer",
            type=gpu.BufferType.Standard,
            format=gpu.Format.R32_UINT,
            element_count=math.ceil(w / 16) * math.ceil(h / 16)
        )

    def update_constant_buffers(self, context):
        context.cmd.begin_marker("Update Constant Buffers")

        super().update_constant_buffers(context)

        # Bin Raster
        round_size = 8 * 64
        min_batches = (1 << 4) * 4
        max_rounds = 64
        batch_size = 1024 # Utility.clamp(context.segment_count / (round_size * min_batches), 1, max_rounds) * round_size

        context.cmd.upload_resource(
            source=np.array([context.segment_count, math.ceil(batch_size), context.w, context.h], dtype='f'),
            destination=self.cb_raster_bin
        )

        context.cmd.end_marker()

    def raster_bin(self, context):
        context.cmd.begin_marker("BinPass")

        context.cmd.dispatch(
            shader=s_raster_bin,

            constants=[
                self.cb_raster_bin
            ],

            inputs=[
                self.b_segment_output,
                self.b_segment_header,
            ],

            outputs=[
                self.b_counters,
                self.b_bin_count
            ],

            x=Budgets.NUM_CU,
        )

        context.cmd.end_marker()

    def raster_coarse(self, context):
        context.cmd.begin_marker("CoarsePass")

        context.cmd.dispatch(
            shader=s_raster_coarse,

            constants=[
            ],

            inputs=[
                self.b_segment_output,
            ],

            x=math.ceil(context.segment_count / 1024),
        )

        context.cmd.end_marker()

    def raster_fine(self, context):
        context.cmd.begin_marker("FinePass")

        context.cmd.dispatch(
            shader=s_raster_fine,

            constants=[
            ],

            inputs=[
                self.b_segment_output,
            ],

            x=math.ceil(context.segment_count / 1024)
        )

        context.cmd.end_marker()

    def clear_buffers(self, context):
        context.cmd.begin_marker("Clear Buffers")

        super().clear_buffers(context)

        Utility.clear_buffer(
            context.cmd,
            0,
            1,
            self.b_counters,
            Utility.ClearMode.UINT
        )

        Utility.clear_buffer(
            context.cmd,
            0,
            math.ceil(context.w / 16) * math.ceil(context.h / 16),
            self.b_bin_count,
            Utility.ClearMode.UINT
        )

        context.cmd.end_marker()

    def go(self, context):
        context.cmd.begin_marker("Raster (Binned)")

        # 1) Dispatch geometry processing and segment setup stages.
        super().go(context)

        # 2) Binning Stage
        self.raster_bin(context)

        # 3) Coarse Stage
        pass

        # 4) Fine Stage
        pass

        context.cmd.end_marker()