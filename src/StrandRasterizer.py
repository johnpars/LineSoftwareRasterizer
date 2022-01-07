import math
import numpy as np
import coalpy.gpu as gpu

from dataclasses import dataclass
from src import Utility
from src import Budgets
from src import StrandDeviceMemory

# Stage Kernels
#############################################################################################################
s_segment_setup = gpu.Shader(file="SegmentSetup.hlsl",     name="SegmentSetup", main_function="SegmentSetup")
s_raster_bin    = gpu.Shader(file="RasterBin.hlsl",        name="RasterBin",    main_function="RasterBin")
s_raster_coarse = gpu.Shader(file="RasterCoarse.hlsl",     name="RasterCoarse", main_function="RasterCoarse")
s_raster_fine   = gpu.Shader(file="RasterFine.hlsl",       name="RasterFine",   main_function="RasterFine")
#############################################################################################################

@dataclass
class Context:
    cmd: gpu.CommandList
    w: int
    h: int
    matrix_v: np.ndarray
    matrix_p: np.ndarray
    strands: StrandDeviceMemory
    strand_count: int
    segment_count: int
    strand_particle_count: int
    target: gpu.Texture


class StrandRasterizer:

    def __init__(self, w, h):

        self.mW = 0
        self.mH = 0

        # Constant Buffers
        self.cb_vertex        = None
        self.cb_segment_setup = None
        self.cb_raster_bin    = None
        self.create_constant_buffers()

        # Resource Buffers
        self.b_segment_count  = None
        self.b_segment_header = None
        self.b_coarse_tile    = None
        self.b_counter        = None
        self.b_ring           = None
        self.create_resource_buffers()

        # Resolution Dependent
        self.b_coarse_tile_count = None
        self.b_coarse_tile_head  = None
        self.update_resolution_dependent_buffers(w, h)

        return

    def create_resource_buffers(self):
        self.b_segment_count = gpu.Buffer(
            name="SegmentCountBuffer",
            type=gpu.BufferType.Standard,
            format=gpu.Format.R32_UINT,
            element_count=Budgets.MAX_SEGMENTS
        )

        self.b_segment_header = gpu.Buffer(
            name="SegmentHeaderBuffer",
            type=gpu.BufferType.Structured,
            stride=Budgets.BYTE_SIZE_SEGMENT_HEADER_FORMAT,
            element_count=Budgets.MAX_SEGMENTS * Budgets.BYTE_SIZE_SEGMENT_HEADER_FORMAT
        )

        self.b_coarse_tile = gpu.Buffer(
            name="CoarseTileSegmentBuffer",
            type=gpu.BufferType.Structured,
            stride=Budgets.BYTE_SIZE_SEGMENT_FORMAT,
            element_count=math.ceil(
                Budgets.BYTE_SIZE_SEGMENT_POOL / Budgets.BYTE_SIZE_SEGMENT_FORMAT)
        )

        self.b_counter = gpu.Buffer(
            name="CounterBuffer",
            type=gpu.BufferType.Standard,
            format=gpu.Format.R32_UINT,
            element_count=1
        )

        self.b_ring = gpu.Buffer(
            name="RingBuffer",
            type=gpu.BufferType.Standard,
            format=gpu.Format.R32_UINT,
            element_count=Budgets.MAX_SEGMENTS
        )

    def create_constant_buffers(self):
        # Vertex Shader
        self.cb_vertex = gpu.Buffer(
            name="ConstantBufferVertex",
            type=gpu.BufferType.Structured,
            stride=((4 * 4) * 4) + ((4 * 4) * 4) + (4 * 4),  # Matrix V, Matrix P, Params
            element_count=1,
            is_constant_buffer=True
        )

        # Segment Setup
        self.cb_segment_setup = gpu.Buffer(
            name="ConstantBufferSegmentSetup",
            type=gpu.BufferType.Structured,
            stride=(4 * 4),
            element_count=1,
            is_constant_buffer=True
        )

        # Raster Bin
        self.cb_raster_bin = gpu.Buffer(
            name="ConstantBufferRasterBin",
            type=gpu.BufferType.Structured,
            stride=(4 * 4),
            element_count=1,
            is_constant_buffer=True
        )

    def update_constant_buffers(self, context):
        context.cmd.begin_marker("UpdateConstantBuffers")

        # Vertex Shader
        context.cmd.upload_resource(
            source=np.array([
                # _MatrixV
                context.matrix_v[0, 0:4],
                context.matrix_v[1, 0:4],
                context.matrix_v[2, 0:4],
                context.matrix_v[3, 0:4],

                # _MatrixP
                context.matrix_p[0, 0:4],
                context.matrix_p[1, 0:4],
                context.matrix_p[2, 0:4],
                context.matrix_p[3, 0:4],

                # _VertexParams
                [context.strand_count, context.strand_particle_count, 0, 0],
            ], dtype='f'),
            destination=self.cb_vertex
        )

        # Segment Setup
        context.cmd.upload_resource(
            source=np.array([context.segment_count, 0, 0, 0], dtype='f'),
            destination=self.cb_segment_setup
        )

        # Bin Raster
        # NOTE: GPU Dependent due to wave size / thread block size.
        round_size = 8 * 64
        min_batches = (1 << 4) * 4
        max_rounds = 64
        batch_size = Utility.clamp(context.segment_count / (round_size * min_batches), 1, max_rounds) * round_size

        context.cmd.upload_resource(
            source=np.array([context.segment_count, batch_size, 0, 0], dtype='f'),
            destination=self.cb_raster_bin
        )

        context.cmd.end_marker()

    def update_resolution_dependent_buffers(self, w, h):
        if w <= self.mW and h <= self.mH:
            return

        self.mW = w
        self.mH = h

        cw = math.ceil(w / Budgets.TILE_SIZE_COARSE)
        ch = math.ceil(h / Budgets.TILE_SIZE_COARSE)

        self.b_coarse_tile_count = gpu.Buffer(
            name="CoarseTileSegmentCount",
            type=gpu.BufferType.Standard,
            format=gpu.Format.R32_UINT,
            element_count=cw * ch
        )

        self.b_coarse_tile_head = gpu.Buffer(
            name="CoarseTileHeadPointerBuffer",
            type=gpu.BufferType.Raw,
            element_count=cw * ch
        )

    def clear_buffers(self, context):
        context.cmd.begin_marker("ClearCoarseBuffers")

        # Tile segment counter buffer
        Utility.clear_buffer(
            context.cmd,
            0,
            math.ceil(context.w / Budgets.TILE_SIZE_COARSE) *
            math.ceil(context.h / Budgets.TILE_SIZE_COARSE),
            self.b_coarse_tile_count
        )

        # Tile head pointer buffer
        Utility.clear_buffer(
            context.cmd,
            -1,  # 0xFFFFFFFF
            math.ceil(context.w / Budgets.TILE_SIZE_COARSE) *
            math.ceil(context.h / Budgets.TILE_SIZE_COARSE),
            self.b_coarse_tile_head
        )

        # Reset the counters
        Utility.clear_buffer(
            context.cmd,
            0,
            1,
            self.b_counter
        )

        Utility.clear_buffer(
            context.cmd,
            0,
            Budgets.MAX_SEGMENTS,
            self.b_ring
        )

        context.cmd.end_marker()

    def segment_setup(self, context):
        context.cmd.begin_marker("SegmentSetupPass")

        groupSize = 512

        context.cmd.dispatch(
            shader=s_segment_setup,

            constants=[
                self.cb_segment_setup,
                self.cb_vertex
            ],

            inputs=[
                context.strands.vertex_buffer,
                context.strands.index_buffer,
                context.strands.strand_buffer
            ],

            outputs=[
                self.b_segment_count,
                self.b_segment_header
            ],

            x=math.ceil(context.segment_count / groupSize),
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
                self.b_segment_count,
            ],

            outputs=[
                self.b_counter,
                self.b_ring
            ],

            x=16,
        )

        context.cmd.end_marker()

    def raster_coarse(self, context):
        context.cmd.begin_marker("CoarsePass")

        context.cmd.dispatch(
            shader=s_raster_coarse,

            constants=[
            ],

            inputs=[
                self.b_segment_count,
            ],

            outputs=[
                self.b_counter
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
                self.b_segment_count,
            ],

            outputs=[
                self.b_counter
            ],

            x=math.ceil(context.segment_count / 1024)
        )

        context.cmd.end_marker()

    def new_frame(self, context):
        self.update_resolution_dependent_buffers(context.w, context.h)
        self.update_constant_buffers(context)
        self.clear_buffers(context)

    def go(self, context):
        context.cmd.begin_marker("Strand Rasterization")

        self.new_frame(context)
        self.segment_setup(context)
        self.raster_bin(context)
        # self.RasterCoarse(context)
        # self.RasterFine(context)

        context.cmd.end_marker()
