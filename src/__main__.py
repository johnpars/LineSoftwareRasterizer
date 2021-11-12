import coalpy.gpu as gpu

from src import Utility
from src import Editor
from src import StrandFactory
from src import StrandDeviceMemory
from src import StrandRasterizer

# Allocate a chunk of device memory resources
deviceMemory = StrandDeviceMemory.StrandDeviceMemory()

# Create a default strand
strands = StrandFactory.Build()

# Layout the initial memory and bind the position data
deviceMemory.Layout(strands.strandCount, strands.strandParticleCount)
deviceMemory.BindStrandPositionData(strands.particlePositions)

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
    StrandRasterizer.Go(
        cmd,
        strands.strandCount,
        strands.strandParticleCount,
        deviceMemory,
        output_target,
        editor.camera.view_matrix,
        editor.camera.proj_matrix,
        w, h
    )

    editor.render_ui(render_args.imgui)

    # Schedule the work.
    gpu.schedule(
        [cmd]
    )


# Invoke the window creation and register our render loop.
window = gpu.Window("StrandRasterizer", 1920, 1080, OnRender)

gpu.run()
