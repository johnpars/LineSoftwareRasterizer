import math
import sys
import numpy as np
import coalpy.gpu as gpu

from dataclasses import dataclass
from src import Utility
from src import Budgets
from src import StrandDeviceMemory

s_vertex_setup  = gpu.Shader(file="VertexSetup.hlsl",  name="VertexSetup",  main_function="VertexSetup")
s_segment_setup = gpu.Shader(file="SegmentSetup.hlsl", name="SegmentSetup", main_function="SegmentSetup")

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
    tesselation: bool
    tesselation_sample_count: int
    oit : bool
    target: gpu.Texture


class Rasterizer:

    def __init__(self, w, h):

        self.mW = 0
        self.mH = 0

        # Constant Buffers
        self.cb_vertex_setup  = None
        self.cb_segment_setup = None
        self.cb_raster_bin    = None
        self.create_constant_buffers()

        # Resource Buffers
        self.b_vertex_output  = None
        self.b_segment_output = None
        self.b_segment_header = None
        self.b_segment_data   = None
        self.create_resource_buffers()

        # Resolution Dependent
        self.update_resolution_dependent_buffers(w, h)

    def create_resource_buffers(self):
        self.b_vertex_output = gpu.Buffer(
            name="VertexOutputBuffer",
            type=gpu.BufferType.Structured,
            stride=Budgets.BYTE_SIZE_VERTEX_OUTPUT_FORMAT,
            element_count=math.ceil(Budgets.BYTE_SIZE_VERTEX_OUTPUT_POOL / Budgets.BYTE_SIZE_VERTEX_OUTPUT_FORMAT)
        )

        self.b_segment_output = gpu.Buffer(
            name="SegmentOutputBuffer",
            type=gpu.BufferType.Raw,
            element_count=Budgets.MAX_SEGMENTS
        )

        self.b_segment_header = gpu.Buffer(
            name="SegmentHeaderBuffer",
            type=gpu.BufferType.Structured,
            stride=Budgets.BYTE_SIZE_SEGMENT_HEADER_FORMAT,
            element_count=Budgets.MAX_SEGMENTS * Budgets.BYTE_SIZE_SEGMENT_HEADER_FORMAT
        )

        self.b_segment_data = gpu.Buffer(
            name="SegmentDataBuffer",
            type=gpu.BufferType.Structured,
            stride=Budgets.BYTE_SIZE_SEGMENT_DATA_FORMAT,
            element_count=Budgets.MAX_SEGMENTS * Budgets.BYTE_SIZE_SEGMENT_DATA_FORMAT
        )

    def create_constant_buffers(self):
        self.cb_vertex_setup = gpu.Buffer(
            name="ConstantBufferVertex",
            type=gpu.BufferType.Structured,
            stride=((4 * 4) * 4) + ((4 * 4) * 4) + (4 * 4),  # Matrix V, Matrix P, Params
            element_count=1,
            is_constant_buffer=True
        )

        self.cb_segment_setup = gpu.Buffer(
            name="ConstantBufferSegmentSetup",
            type=gpu.BufferType.Structured,
            stride=(4 * 4),
            element_count=1,
            is_constant_buffer=True
        )

    def update_constant_buffers(self, context):
        # Vertex Setup
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

            destination=self.cb_vertex_setup
        )

        # Segment Setup
        context.cmd.upload_resource(
            source=np.array([context.segment_count, 0, 0, 0], dtype='f'),
            destination=self.cb_segment_setup
        )

    def update_resolution_dependent_buffers(self, w, h):
        if w <= self.mW and h <= self.mH:
            return

        self.mW = w
        self.mH = h

    def clear_buffers(self, context):
        pass

    def vertex_setup(self, context):
        context.cmd.begin_marker("VertexSetupPass")

        # Compute the vertex count to determine the launch size
        vertex_count = context.strand_particle_count * context.strand_count

        context.cmd.dispatch(
            shader=s_vertex_setup,

            constants=[
                self.cb_vertex_setup
            ],

            inputs=[
                context.strands.b_vertices,
                context.strands.b_strands
            ],

            outputs=[
                self.b_vertex_output
            ],

            x=math.ceil(vertex_count / Budgets.NUM_LANE_PER_WAVE)
        )

        context.cmd.end_marker()

    def segment_setup(self, context):
        context.cmd.begin_marker("SegmentSetupPass")

        groupSize = 512

        context.cmd.dispatch(
            shader=s_segment_setup,

            constants=[
                self.cb_segment_setup
            ],

            inputs=[
                self.b_vertex_output,
                context.strands.b_indices,
            ],

            outputs=[
                self.b_segment_output,
                self.b_segment_header,
                self.b_segment_data
            ],

            x=math.ceil(context.segment_count / groupSize),
        )

        context.cmd.end_marker()

    def new_frame(self, context):
        self.update_resolution_dependent_buffers(context.w, context.h)
        self.update_constant_buffers(context)
        self.clear_buffers(context)

    def go(self, context):
        self.new_frame(context)
        self.vertex_setup(context)
        self.segment_setup(context)
