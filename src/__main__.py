import coalpy.gpu as gpu
import os

from src import Utility
from src import Editor
from src import Debug
from src import StrandFactory
from src import StrandDeviceMemory
from src import StrandRasterizer

initialWidth  = 1920
initialHeight = 1080

# Allocate a chunk of device memory resources
deviceMemory = StrandDeviceMemory.StrandDeviceMemory()

# Create a default strand
# strands = StrandFactory.BuildProcedural()
strands = StrandFactory.BuildFromAsset("cube_hair")

# Layout the initial memory and bind the position data
deviceMemory.Layout(strands.strandCount, strands.strandParticleCount)
deviceMemory.BindStrandPositionData(strands.particlePositions)

# Create the rasterizer
rasterizer = StrandRasterizer.StrandRasterizer(initialWidth, initialHeight)

editor = Editor.Editor(deviceMemory, strands)

def OnRender(render_args: gpu.RenderArgs):
    output_target = render_args.window.display_texture

    w = render_args.width
    h = render_args.height

    # Process user input and interface
    editor.UpdateCamera(w, h, render_args.delta_time, render_args.window)

    cmd = gpu.CommandList()

    Utility.ClearTarget(
        cmd,
        [0.0, 0.0, 0.0, 0.0],
        output_target, w, h
    )

    # Draw the strands.
    rasterizer.BruteForce(
        cmd,
        editor.m_strands.strandCount,
        editor.m_strands.strandParticleCount,
        deviceMemory,
        output_target,
        editor.camera.view_matrix,
        editor.camera.proj_matrix,
        w, h
    )

#    rasterizer.UpdateResolutionDependentBuffers(w, h)
#
#    # Invoke the coarse rasterizer, binning segments into tiles.
#    rasterizer.CoarsePass(
#        cmd,
#        editor.m_strands.strandCount,
#        deviceMemory,
#        w, h
#    )
#
#    # Debug view the results of the coarse rasterization pass.
#    Debug.SegmentsPerTile(
#        cmd,
#        output_target,
#        w, h,
#        rasterizer
#    )

    # editor.render_ui(render_args.imgui)

    # Schedule the work.
    gpu.schedule(
        [cmd]
    )


# Invoke the window creation and register our render loop.
window = gpu.Window("StrandRasterizer", initialWidth, initialHeight, OnRender)

gpu.run()
