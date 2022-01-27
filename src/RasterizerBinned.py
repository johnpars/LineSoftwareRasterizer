import math
import numpy as np
import coalpy.gpu as gpu

from src import Utility
from src import Budgets
from src import PrefixSum
from src import Rasterizer

# Stage Kernels
s_raster_bin            = gpu.Shader(file="RasterBin.hlsl",     name="RasterBin",     main_function="RasterBin")
s_raster_bin_tes        = gpu.Shader(file="RasterBin.hlsl",     name="RasterBin",     main_function="RasterBin", defines=["RASTER_CURVE"])
s_raster_fine           = gpu.Shader(file="RasterFine.hlsl",    name="RasterFine",    main_function="RasterFine")
s_raster_fine_tes       = gpu.Shader(file="RasterFine.hlsl",    name="RasterFine",    main_function="RasterFine", defines=["RASTER_CURVE"])
s_raster_fine_oit       = gpu.Shader(file="RasterFineOIT.hlsl", name="RasterFineOIT", main_function="RasterFineOIT")
s_build_work_queue_args = gpu.Shader(file="WorkQueue.hlsl",     name="WorkQueueArgs", main_function="BuildWorkQueueArgs")
s_build_work_queue      = gpu.Shader(file="WorkQueue.hlsl",     name="WorkQueue",     main_function="BuildWorkQueue")


class RasterizerBinned(Rasterizer.Rasterizer):
    def __init__(self, w, h):

        self.bin_w = math.ceil(w / Budgets.TILE_SIZE_BIN)
        self.bin_h = math.ceil(h / Budgets.TILE_SIZE_BIN)

        # Resources
        self.b_bin_records = None
        self.b_bin_records_counter = None
        self.b_bin_counters = None
        self.b_bin_min_z = None
        self.b_bin_max_z = None
        self.b_bin_offsets = None
        self.b_work_queue = None
        self.b_work_queue_args = None
        self.b_prefix_sum_args = None

        # Constant buffers
        self.cb_raster_bin = None
        self.cb_raster_fine = None

        # Will invoke the creation of resources
        super().__init__(w, h)

    def create_resource_buffers(self):
        super().create_resource_buffers()

        self.b_bin_records = gpu.Buffer(
            name="BinRecords",
            type=gpu.BufferType.Structured,
            stride=Budgets.BYTE_SIZE_BIN_RECORD_FORMAT,
            element_count=math.ceil(Budgets.BYTE_SIZE_BIN_RECORD_POOL / Budgets.BYTE_SIZE_BIN_RECORD_FORMAT)
        )

        self.b_bin_records_counter = gpu.Buffer(
            name="BinRecordsCounter",
            type=gpu.BufferType.Standard,
            format=gpu.Format.R32_UINT,
            element_count=1
        )

        self.b_work_queue = gpu.Buffer(
            name="WorkQueue",
            type=gpu.BufferType.Standard,
            format=gpu.Format.R32_UINT,
            element_count=math.ceil(Budgets.BYTE_SIZE_WORK_QUEUE_POOL / Budgets.BYTE_SIZE_WORK_QUEUE_FORMAT)
        )

        self.b_work_queue_args = gpu.Buffer(
            name="WorkQueueArgs",
            type=gpu.BufferType.Standard,
            format=gpu.Format.RGBA_32_UINT,
            element_count=1
        )

    def create_constant_buffers(self):
        super().create_constant_buffers()

        self.cb_raster_bin = gpu.Buffer(
            name="ConstantBufferRasterBin",
            type=gpu.BufferType.Structured,
            stride=(4 * 4) * 2,
            element_count=1,
            is_constant_buffer=True
        )

        self.cb_raster_fine = gpu.Buffer(
            name="ConstantBufferRasterFine",
            type=gpu.BufferType.Structured,
            stride=(4 * 4) * 2,
            element_count=1,
            is_constant_buffer=True
        )

    def update_resolution_dependent_buffers(self, w, h):
        if w <= self.mW and h <= self.mH:
            return

        self.bin_w = math.ceil(w / Budgets.TILE_SIZE_BIN)
        self.bin_h = math.ceil(h / Budgets.TILE_SIZE_BIN)

        super().update_resolution_dependent_buffers(w, h)

        self.b_bin_counters = gpu.Buffer(
            name="BinCountBuffer",
            type=gpu.BufferType.Standard,
            format=gpu.Format.R32_UINT,
            element_count=self.bin_w * self.bin_h
        )

        self.b_bin_min_z = gpu.Buffer(
            name="BinMinZ",
            type=gpu.BufferType.Standard,
            format=gpu.Format.R32_UINT,
            element_count=self.bin_w * self.bin_h
        )

        self.b_bin_max_z = gpu.Buffer(
            name="BinMaxZ",
            type=gpu.BufferType.Standard,
            format=gpu.Format.R32_UINT,
            element_count=self.bin_w * self.bin_h
        )

        self.b_prefix_sum_args = PrefixSum.allocate_args(self.bin_w * self.bin_h)

    def update_constant_buffers(self, context):
        context.cmd.begin_marker("Update Constant Buffers")

        super().update_constant_buffers(context)

        context.cmd.upload_resource(
            source=np.array([
                context.segment_count,
                context.w,
                context.h,
                Budgets.TILE_SIZE_BIN,
                self.bin_w,
                self.bin_h,
                context.tesselation_sample_count

            ], dtype='f'),
            destination=self.cb_raster_bin
        )

        context.cmd.upload_resource(
            source=np.array([
                context.w,
                context.h,
                Budgets.TILE_SIZE_BIN,
                self.bin_w,
                self.bin_h,
                context.tesselation_sample_count
            ], dtype='f'),
            destination=self.cb_raster_fine
        )

        context.cmd.end_marker()

    def clear_buffers(self, context):
        context.cmd.begin_marker("Clear Buffers")

        super().clear_buffers(context)

        Utility.clear_buffer(
            context.cmd,
            0,
            1,
            self.b_bin_records_counter,
            Utility.ClearMode.UINT
        )

        Utility.clear_buffer(
            context.cmd,
            0,
            self.bin_w * self.bin_h,
            self.b_bin_counters,
            Utility.ClearMode.UINT
        )

        Utility.clear_buffer(
            context.cmd,
            (1 << 31) - 1, # Max int
            self.bin_w * self.bin_h,
            self.b_bin_min_z,
            Utility.ClearMode.UINT
        )

        Utility.clear_buffer(
            context.cmd,
            0,
            self.bin_w * self.bin_h,
            self.b_bin_max_z,
            Utility.ClearMode.UINT
        )

        context.cmd.end_marker()

    def raster_bin(self, context):
        context.cmd.begin_marker("BinPass")

        context.cmd.dispatch(
            shader=s_raster_bin_tes if context.tesselation else s_raster_bin,

            constants=[
                self.cb_raster_bin
            ],

            inputs=[
                self.b_segment_output,
                self.b_segment_data,
                self.b_vertex_output,
                self.b_segment_header
            ],

            outputs=[
                self.b_bin_records,
                self.b_bin_records_counter,
                self.b_bin_counters,
                self.b_bin_min_z,
                self.b_bin_max_z
            ],

            x=math.ceil(context.segment_count / Budgets.NUM_LANE_PER_WAVE)
        )

        context.cmd.end_marker()
        
    def build_work_queue(self, context):

        context.cmd.begin_marker("BuildWorkQueue")

        # 1) Generate offset indices into the global work queue, by performing a prefix sum on the bin counters.
        self.b_bin_offsets = PrefixSum.run(
            context.cmd,
            self.b_bin_counters,
            self.b_prefix_sum_args,
            True,
            self.bin_w * self.bin_h
        )

        # 2) Derive a dispatch launch size from the amount of bin records.
        context.cmd.dispatch(
            shader=s_build_work_queue_args,
            inputs=[
                self.b_bin_records_counter
            ],
            outputs=[
                self.b_work_queue_args
            ],
            x=1
        )

        # 3) Indirectly dispatch the work queue construction.
        context.cmd.dispatch(
            indirect_args=self.b_work_queue_args,
            shader=s_build_work_queue,
            inputs=[
                self.b_bin_offsets,
                self.b_bin_records,
                self.b_bin_records_counter
            ],
            outputs=[
                self.b_work_queue
            ]
        )

        context.cmd.end_marker()

    def raster_fine(self, context):
        context.cmd.begin_marker("FinePass")

        if context.oit:
            shader = None if context.tesselation else s_raster_fine_oit
        else:
            shader = s_raster_fine_tes if context.tesselation else s_raster_fine

        if shader is None:
            print("OIT + Tesselation not yet implemented.")
            return

        context.cmd.dispatch(
            shader=shader,

            constants=[
                self.cb_raster_fine
            ],

            inputs=[
                self.b_work_queue,
                self.b_bin_offsets,
                self.b_bin_counters,
                self.b_segment_data,
                self.b_vertex_output,
                self.b_bin_min_z,
                self.b_bin_max_z
            ],

            outputs=[
                context.target
            ],

            x=self.bin_w,
            y=self.bin_h
        )

        context.cmd.end_marker()

    def go(self, context):
        context.cmd.begin_marker("Raster (Binned)")

        # 1) Dispatch geometry processing and segment setup stages.
        super().go(context)

        # 2) Binning Stage
        self.raster_bin(context)

        # 3) Work Queue
        self.build_work_queue(context)

        # 4) Fine Stage
        self.raster_fine(context)

        context.cmd.end_marker()
