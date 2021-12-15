import math
import numpy as np
import coalpy.gpu as gpu

from dataclasses import dataclass

from src import Utility
from src import StrandDeviceMemory

ShaderBruteForce       = gpu.Shader(file ="StrandRasterizer.hlsl", name ="BruteForce",       main_function ="BruteForce")
ShaderSegmentSetupPass = gpu.Shader(file ="SegmentSetup.hlsl", name ="SegmentSetup", main_function ="SegmentSetup")
ShaderCoarsePass       = gpu.Shader(file ="StrandRasterizer.hlsl", name ="CoarsePass",       main_function ="CoarsePass")
ShaderFinePass         = gpu.Shader(file ="StrandRasterizer.hlsl", name ="FinePass",         main_function ="FinePass")

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
    SegmentBufferPoolByteSize   = 1024 * 1024 * 1024     # 128mb
    SegmentBufferFormatByteSize = (4 * 4) + (4 * 4) + 4 + 4 # P0, P1, Segment Index, Next

    CoarseTileSize = 16

    def __init__(self, w, h):

        self.mSegmentBuffer = gpu.Buffer(
            name = "SegmentBuffer",
            type = gpu.BufferType.Structured,
            stride = StrandRasterizer.SegmentBufferFormatByteSize,
            element_count = math.ceil(StrandRasterizer.SegmentBufferPoolByteSize / StrandRasterizer.SegmentBufferFormatByteSize)
        )

        # Create the tile segment data buffer.
        self.mCoarseTileSegmentBuffer = gpu.Buffer(
            name = "CoarseTileSegmentBuffer",
            type = gpu.BufferType.Structured,
            stride = StrandRasterizer.SegmentBufferFormatByteSize,
            element_count = math.ceil(StrandRasterizer.SegmentBufferPoolByteSize / StrandRasterizer.SegmentBufferFormatByteSize)
        )

        self.mCoarseTileSegmentCounter = gpu.Buffer(
            name = "CoarseTileSegmentCounter",
            type = gpu.BufferType.Standard,
            format = gpu.Format.R32_UINT,
            element_count = 1
        )

        self.CreateConstantBuffers()

        self.mW = 0
        self.mH = 0
        self.UpdateResolutionDependentBuffers(w, h)

        return

    def CreateConstantBuffers(self):

        # Segment Setup
        self.mConstantBufferSegmentSetup = gpu.Buffer(
            name = "ConstantBufferSegmentSetup",
            type = gpu.BufferType.Structured,
            stride = (4 * 4),
            element_count = 1,
            is_constant_buffer = True
        )

        # Vertex Shader
        self.mConstantBufferVertex = gpu.Buffer(
            name = "ConstantBufferVertex",
            type = gpu.BufferType.Structured,
            stride = ((4 * 4) * 4) + ((4 * 4) * 4) + (4 * 4), # Matrix V, Matrix P, Params
            element_count = 1,
            is_constant_buffer = True
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
            source = np.array([context.segmentCount, 0, 0, 0], dtype='f'),
            destination = self.mConstantBufferSegmentSetup
        )

        context.cmd.end_marker()

    def UpdateResolutionDependentBuffers(self, w, h):

        if (w <= self.mW and h <= self.mH):
            return

        self.mW = w
        self.mH = h

        cW = math.ceil(w / StrandRasterizer.CoarseTileSize)
        cH = math.ceil(h / StrandRasterizer.CoarseTileSize)

        self.mCoarseTileSegmentCount = gpu.Buffer(
            name = "CoarseTileSegmentCount",
            type = gpu.BufferType.Standard,
            format = gpu.Format.R32_UINT,
            element_count = cW * cH
        )

        self.mCoarseTileHeadPointerBuffer = gpu.Buffer(
            name = "CoarseTileHeadPointerBuffer",
            type = gpu.BufferType.Raw,
            element_count = cW * cH
        )

    def BruteForce(self, cmd, strandCount, strandParticleCount, strands : StrandDeviceMemory, target : gpu.Texture, matrixV, matrixP, w, h):

        cmd.dispatch(
            shader = ShaderBruteForce,

            constants = np.array([
                # _MatrixV
                matrixV[0, 0:4],
                matrixV[1, 0:4],
                matrixV[2, 0:4],
                matrixV[3, 0:4],

                # _MatrixP
                matrixP[0, 0:4],
                matrixP[1, 0:4],
                matrixP[2, 0:4],
                matrixP[3, 0:4],

                # _ScreenParams
                [ w, h, 1.0 / w, 1.0 / h ],

                # _Params0
                [ strandCount, strandParticleCount, 0.0, 0.0 ]
            ], dtype='f'),

            x = math.ceil(w / 8),
            y = math.ceil(h / 8),

            inputs = [
                strands.mVertexBuffer,
                strands.mIndexBuffer,
                strands.mStrandDataBuffer
            ],

            outputs = target
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
            -1, # 0xFFFFFFFF
            math.ceil(context.w / StrandRasterizer.CoarseTileSize) *
            math.ceil(context.h / StrandRasterizer.CoarseTileSize),
            self.mCoarseTileHeadPointerBuffer
        )

        # Reset the counter
        Utility.ClearBuffer(
            context.cmd,
            0,
            1,
            self.mCoarseTileSegmentCounter
        )

        context.cmd.end_marker()

    def SegmentSetupPass(self, context):
        context.cmd.begin_marker("SegmentSetupPass")

        context.cmd.dispatch(
            shader = ShaderSegmentSetupPass,

            constants=[
                self.mConstantBufferSegmentSetup,
                self.mConstantBufferVertex
            ],

            inputs = [
                context.strands.mVertexBuffer,
                context.strands.mIndexBuffer,
                context.strands.mStrandDataBuffer
            ],

            outputs = [
                self.mSegmentBuffer
            ],

            x = math.ceil(context.segmentCount / 1024),
            y = 1,
            z = 1
        )

        context.cmd.end_marker()

    def CoarsePassPersistent(self, context):
        return

    def CoarsePass(self, cmd, strandCount, strandParticleCount, strands : StrandDeviceMemory, matrixV, matrixP, w, h):
        cmd.begin_marker("CoarsePass")

        groupDimX = math.ceil(w / StrandRasterizer.CoarseTileSize)
        groupDimY = math.ceil(h / StrandRasterizer.CoarseTileSize)

        # Compute segment count
        segmentCount = strandCount * (strandParticleCount - 1)

        cmd.dispatch(
            shader = ShaderCoarsePass,

            constants = np.array([
                # _MatrixV
                matrixV[0, 0:4],
                matrixV[1, 0:4],
                matrixV[2, 0:4],
                matrixV[3, 0:4],

                # _MatrixP
                matrixP[0, 0:4],
                matrixP[1, 0:4],
                matrixP[2, 0:4],
                matrixP[3, 0:4],

                # _ScreenParams
                [w, h, 1.0 / w, 1.0 / h],

                # _Params0
                [ groupDimX, groupDimY, StrandRasterizer.CoarseTileSize, segmentCount ]
            ], dtype='f'),

            inputs = [
                strands.mVertexBuffer,
                strands.mIndexBuffer,
                strands.mStrandDataBuffer
            ],

            outputs = [
                self.mCoarseTileSegmentCount,
                self.mCoarseTileHeadPointerBuffer,
                self.mCoarseTileSegmentBuffer,
                self.mCoarseTileSegmentCounter
            ],

            x = math.ceil(segmentCount / StrandRasterizer.CoarseTileSize),
            y = 1,
            z = 1
        )

        cmd.end_marker()

        return

    def FinePass(self, cmd, strandCount, strandParticleCount, target : gpu.Texture, w, h):
        cmd.begin_marker("FinePass")

        groupDimX = math.ceil(w / StrandRasterizer.CoarseTileSize)
        groupDimY = math.ceil(h / StrandRasterizer.CoarseTileSize)

        cmd.dispatch(
            shader = ShaderFinePass,

            constants = np.array([
                # _ScreenParams
                [w, h, 1.0 / w, 1.0 / h],
                [groupDimX, groupDimY, strandCount, strandParticleCount]
            ], dtype='f'),

            inputs = [
                self.mCoarseTileHeadPointerBuffer,
                self.mCoarseTileSegmentBuffer,
            ],

            outputs = target,

            x = math.ceil(w / StrandRasterizer.CoarseTileSize),
            y = math.ceil(h / StrandRasterizer.CoarseTileSize),
            z = 1
        )

        cmd.end_marker()
        return

    def NewFrame(self, context):
        self.UpdateResolutionDependentBuffers(context.w, context.h)
        self.UpdateConstantBuffers(context)
        self.ClearCoarsePassBuffers(context)

    def Go(self, context):

        # 1) Preliminary Rasterization Setup
        self.NewFrame(context)

        # 2) Segment Setup
        self.SegmentSetupPass(context)

        # 3) Coarse Raster

        # 4) Fine Raster

        return