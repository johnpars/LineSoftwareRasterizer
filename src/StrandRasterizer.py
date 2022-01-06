import math
import numpy as np
import coalpy.gpu as gpu

from dataclasses import dataclass

from src import Utility
from src import StrandDeviceMemory

S_SegmentSetup = gpu.Shader(file="SegmentSetup.hlsl", name="SegmentSetup", main_function="SegmentSetup")
S_RasterBin    = gpu.Shader(file="RasterBin.hlsl",    name="RasterBin",    main_function="RasterBin")
S_RasterCoarse = gpu.Shader(file="RasterCoarse.hlsl", name="RasterCoarse", main_function="RasterCoarse")
S_RasterFine   = gpu.Shader(file="RasterFine.hlsl",   name="RasterFine",   main_function="RasterFine")
S_RasterBrute  = gpu.Shader(file="StrandRasterizer.hlsl",   name="BruteForce",   main_function="BruteForce")


# Container for various contextual frame information.
@dataclass
class Context:
    cmd: gpu.CommandList
    w: int
    h: int
    matrixV: np.ndarray
    matrixP: np.ndarray
    strands: StrandDeviceMemory
    strandCount: int
    segmentCount: int
    strandParticleCount: int
    target: gpu.Texture


class StrandRasterizer:

    # Tile Segment Buffer
    SegmentBufferPoolByteSize = 32 * 1024 * 1024  # 1GB
    SegmentBufferFormatByteSize = (4 * 4) + (4 * 4) + 4 + 4  # P0, P1, Segment Index, Next

    MaxSegments = 1 << 22
    SegmentHeaderFormatByteSize = (4 * 4)

    CoarseTileSize = 16

    def __init__(self, w, h):
        self.mSegmentCountBuffer = gpu.Buffer(
            name="SegmentCountBuffer",
            type=gpu.BufferType.Standard,
            format=gpu.Format.R32_UINT,
            element_count=StrandRasterizer.MaxSegments
        )

        self.mSegmentHeaderBuffer = gpu.Buffer(
            name="SegmentHeaderBuffer",
            type=gpu.BufferType.Structured,
            stride=StrandRasterizer.SegmentHeaderFormatByteSize,
            element_count=StrandRasterizer.MaxSegments * StrandRasterizer.SegmentHeaderFormatByteSize
        )

        # Create the tile segment data buffer.
        self.mCoarseTileSegmentBuffer = gpu.Buffer(
            name="CoarseTileSegmentBuffer",
            type=gpu.BufferType.Structured,
            stride=StrandRasterizer.SegmentBufferFormatByteSize,
            element_count=math.ceil(
                StrandRasterizer.SegmentBufferPoolByteSize / StrandRasterizer.SegmentBufferFormatByteSize)
        )

        self.mCounterBuffer = gpu.Buffer(
            name="CounterBuffer",
            type=gpu.BufferType.Standard,
            format=gpu.Format.R32_UINT,
            element_count=1
        )

        # Temp Ring Buffer view
        self.mRingBuffer = gpu.Buffer(
            name="RingBuffer",
            type=gpu.BufferType.Standard,
            format=gpu.Format.R32_UINT,
            element_count=StrandRasterizer.MaxSegments
        )

        self.CreateConstantBuffers()

        self.mW = 0
        self.mH = 0
        self.UpdateResolutionDependentBuffers(w, h)

        return

    def CreateConstantBuffers(self):
        # Vertex Shader
        self.mConstantBufferVertex = gpu.Buffer(
            name="ConstantBufferVertex",
            type=gpu.BufferType.Structured,
            stride=((4 * 4) * 4) + ((4 * 4) * 4) + (4 * 4),  # Matrix V, Matrix P, Params
            element_count=1,
            is_constant_buffer=True
        )

        # Segment Setup
        self.mConstantBufferSegmentSetup = gpu.Buffer(
            name="ConstantBufferSegmentSetup",
            type=gpu.BufferType.Structured,
            stride=(4 * 4),
            element_count=1,
            is_constant_buffer=True
        )

        # Raster Bin
        self.mConstantBufferRasterBin = gpu.Buffer(
            name="ConstantBufferRasterBin",
            type=gpu.BufferType.Structured,
            stride=(4 * 4),
            element_count=1,
            is_constant_buffer=True
        )

    def UpdateConstantBuffers(self, context):
        context.cmd.begin_marker("UpdateConstantBuffers")

        # Vertex Shader
        context.cmd.upload_resource(
            source=np.array([
                # _MatrixV
                context.matrixV[0, 0:4],
                context.matrixV[1, 0:4],
                context.matrixV[2, 0:4],
                context.matrixV[3, 0:4],

                # _MatrixP
                context.matrixP[0, 0:4],
                context.matrixP[1, 0:4],
                context.matrixP[2, 0:4],
                context.matrixP[3, 0:4],

                # _VertexParams
                [context.strandCount, context.strandParticleCount, 0, 0],
            ], dtype='f'),
            destination=self.mConstantBufferVertex
        )

        # Segment Setup
        context.cmd.upload_resource(
            source=np.array([context.segmentCount, 0, 0, 0], dtype='f'),
            destination=self.mConstantBufferSegmentSetup
        )

        # Bin Raster
        # NOTE: GPU Dependent due to wave size / thread block size.
        roundSize = 8 * 64
        minBatches = (1 << 4) * 4
        maxRounds = 64
        batchSize = Utility.Clamp(context.segmentCount / (roundSize * minBatches), 1, maxRounds) * roundSize

        context.cmd.upload_resource(
            source=np.array([context.segmentCount, batchSize, 0, 0], dtype='f'),
            destination=self.mConstantBufferRasterBin
        )

        context.cmd.end_marker()

    def UpdateResolutionDependentBuffers(self, w, h):
        if w <= self.mW and h <= self.mH:
            return

        self.mW = w
        self.mH = h

        cW = math.ceil(w / StrandRasterizer.CoarseTileSize)
        cH = math.ceil(h / StrandRasterizer.CoarseTileSize)

        self.mCoarseTileSegmentCount = gpu.Buffer(
            name="CoarseTileSegmentCount",
            type=gpu.BufferType.Standard,
            format=gpu.Format.R32_UINT,
            element_count=cW * cH
        )

        self.mCoarseTileHeadPointerBuffer = gpu.Buffer(
            name="CoarseTileHeadPointerBuffer",
            type=gpu.BufferType.Raw,
            element_count=cW * cH
        )

    def ClearCoarsePassBuffers(self, context):
        context.cmd.begin_marker("ClearCoarseBuffers")

        # Tile segment counter buffer
        Utility.ClearBuffer(
            context.cmd,
            0,
            math.ceil(context.w / StrandRasterizer.CoarseTileSize) *
            math.ceil(context.h / StrandRasterizer.CoarseTileSize),
            self.mCoarseTileSegmentCount
        )

        # Tile head pointer buffer
        Utility.ClearBuffer(
            context.cmd,
            -1,  # 0xFFFFFFFF
            math.ceil(context.w / StrandRasterizer.CoarseTileSize) *
            math.ceil(context.h / StrandRasterizer.CoarseTileSize),
            self.mCoarseTileHeadPointerBuffer
        )

        # Reset the counters
        Utility.ClearBuffer(
            context.cmd,
            0,
            1,
            self.mCounterBuffer
        )

        Utility.ClearBuffer(
            context.cmd,
            0,
            StrandRasterizer.MaxSegments,
            self.mRingBuffer
        )

        context.cmd.end_marker()

    def SegmentSetup(self, context):
        context.cmd.begin_marker("SegmentSetupPass")

        groupSize = 512

        context.cmd.dispatch(
            shader=S_SegmentSetup,

            constants=[
                self.mConstantBufferSegmentSetup,
                self.mConstantBufferVertex
            ],

            inputs=[
                context.strands.mVertexBuffer,
                context.strands.mIndexBuffer,
                context.strands.mStrandDataBuffer
            ],

            outputs=[
                self.mSegmentCountBuffer,
                self.mSegmentHeaderBuffer
            ],

            x=math.ceil(context.segmentCount / groupSize),
        )

        context.cmd.end_marker()

    def RasterBin(self, context):
        context.cmd.begin_marker("BinPass")

        context.cmd.dispatch(
            shader=S_RasterBin,

            constants=[
                self.mConstantBufferRasterBin
            ],

            inputs=[
                self.mSegmentCountBuffer,
            ],

            outputs=[
                self.mCounterBuffer,
                self.mRingBuffer
            ],

            x=16,
        )

        context.cmd.end_marker()

    def RasterCoarse(self, context):
        context.cmd.begin_marker("CoarsePass")

        context.cmd.dispatch(
            shader=S_RasterCoarse,

            constants=[
            ],

            inputs=[
                self.mSegmentCountBuffer,
            ],

            outputs=[
                self.mCounterBuffer
            ],

            x=math.ceil(context.segmentCount / 1024),
        )

        context.cmd.end_marker()

    def RasterFine(self, context):
        context.cmd.begin_marker("FinePass")

        context.cmd.dispatch(
            shader=S_RasterFine,

            constants=[
            ],

            inputs=[
                self.mSegmentCountBuffer,
            ],

            outputs=[
                self.mCounterBuffer
            ],

            x=math.ceil(context.segmentCount / 1024)
        )

        context.cmd.end_marker()

    @staticmethod
    def RasterBruteForce(context):
        context.cmd.begin_marker("BruteForcePass")

        context.cmd.dispatch(
            shader=S_RasterBrute,

            constants=np.array([
                # _MatrixV
                context.matrixV[0, 0:4],
                context.matrixV[1, 0:4],
                context.matrixV[2, 0:4],
                context.matrixV[3, 0:4],

                # _MatrixP
                context.matrixP[0, 0:4],
                context.matrixP[1, 0:4],
                context.matrixP[2, 0:4],
                context.matrixP[3, 0:4],

                # _ScreenParams
                [context.w, context.h, 1.0 / context.w, 1.0 / context.h],

                # _Params0
                [context.strandCount, context.strandParticleCount, 0, 0],
            ], dtype='f'),

            inputs=[
                context.strands.mVertexBuffer,
                context.strands.mIndexBuffer,
                context.strands.mStrandDataBuffer
            ],

            outputs=[
                context.target
            ],

            x=math.ceil(context.w / 8),
            y=math.ceil(context.h / 8),
        )

        context.cmd.end_marker()

    def NewFrame(self, context):
        self.UpdateResolutionDependentBuffers(context.w, context.h)
        self.UpdateConstantBuffers(context)
        self.ClearCoarsePassBuffers(context)

    def Go(self, context):
        context.cmd.begin_marker("Strand Rasterization")

        self.NewFrame(context)
        self.SegmentSetup(context)
        self.RasterBin(context)
        # self.RasterCoarse(context)
        # self.RasterFine(context)

        context.cmd.end_marker()
