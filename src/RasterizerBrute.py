import math
import numpy as np
import coalpy.gpu as gpu

from src import Utility
from src import Budgets
from src import Rasterizer

s_raster_coverage = gpu.Shader(file="brute/RasterCoverage.hlsl", name="RasterCoverage", main_function="RasterCoverage")
s_raster_resolve  = gpu.Shader(file="brute/RasterResolve.hlsl",  name="RasterResolve",  main_function="RasterResolve")


class RasterizerBrute(Rasterizer.Rasterizer):

    def __init__(self, w, h):

        # Resources
        self.b_fragment_counter = None
        self.b_fragment_data = None
        self.b_head_pointer = None

        # Constant buffers
        self.cb_brute = None

        super().__init__(w, h)

    def create_resource_buffers(self):
        super().create_resource_buffers()

        self.b_fragment_counter = gpu.Buffer(
            name="FragmentCounterBuffer",
            type=gpu.BufferType.Standard,
            element_count=1
        )

        self.b_fragment_data = gpu.Buffer(
            name="FragmentDataBuffer",
            type=gpu.BufferType.Structured,
            stride=Budgets.BYTE_SIZE_FRAGMENT_DATA_FORMAT,
            element_count=math.ceil(Budgets.BYTE_SIZE_FRAGMENT_DATA_POOL / Budgets.BYTE_SIZE_FRAGMENT_DATA_FORMAT)
        )

    def update_resolution_dependent_buffers(self, w, h):

        if w <= self.mW and h <= self.mH:
            return

        super().update_resolution_dependent_buffers(w, h)

        self.b_head_pointer = gpu.Buffer(
            name="HeadPointerBuffer",
            type=gpu.BufferType.Raw,
            element_count=w*h
        )

    def create_constant_buffers(self):
        super().create_constant_buffers()

        self.cb_brute = gpu.Buffer(
            name="ConstantBufferBrute",
            type=gpu.BufferType.Structured,
            stride=(4 * 4),
            element_count=1,
            is_constant_buffer=True
        )

    def update_constant_buffers(self, context):
        super().update_constant_buffers(context)

        context.cmd.upload_resource(
            source=np.array([context.segment_count, context.w, context.h, 0], dtype='f'),
            destination=self.cb_brute
        )

    def clear_buffers(self, context):
        super().clear_buffers(context)

        Utility.clear_buffer(
            context.cmd,
            0,
            1,
            self.b_fragment_counter
        )

        Utility.clear_buffer(
            context.cmd,
            -1,  # 0xFFFFFFFF
            context.w * context.h,
            self.b_head_pointer
        )

    def raster_coverage(self, context):

        context.cmd.begin_marker("Coverage")

        context.cmd.dispatch(
            shader=s_raster_coverage,

            constants=[
                self.cb_brute
            ],

            inputs=[
                self.b_segment_output,
                self.b_segment_header,
                self.b_segment_data,
                self.b_vertex_output
            ],

            outputs=[
                self.b_fragment_counter,
                self.b_head_pointer,
                self.b_fragment_data,
            ],

            x=math.ceil(context.segment_count / Budgets.NUM_LANE_PER_WAVE)
        )

        context.cmd.end_marker()

    def raster_resolve(self, context):
        context.cmd.begin_marker("Resolve")

        context.cmd.dispatch(
            shader=s_raster_resolve,

            constants=[
                self.cb_brute
            ],

            inputs=[
                self.b_head_pointer,
                self.b_fragment_data,
            ],

            outputs=[
                context.target
            ],

            x=math.ceil(context.w / 8),
            y=math.ceil(context.h / 8)
        )

        context.cmd.end_marker()

    def go(self, context):
        context.cmd.begin_marker("Raster (Brute)")

        # 1) Dispatch geometry processing and segment setup stages (clipping, etc).
        super().go(context)

        # 2) Coverage pass that pushes all contributing fragments into a linked list.
        self.raster_coverage(context)

        # 3) Resolve pass that walks down the fragment linked list for each pixel and solves transmittance function.
        self.raster_resolve(context)

        context.cmd.end_marker()