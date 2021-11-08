import coalpy.gpu as gpu

from src import Utility
from src import Editor
from src import StrandGroup
from src import StrandGroupGPU
from src import StrandRasterizer

# Create the dev strand (low resolution so that we can demonstrate tesselation)
strands = StrandGroupGPU.StrandGroupGPU()
strands.CreateSimpleStrand()

editor = Editor.Editor()

def OnRender(render_args: gpu.RenderArgs):
    output_target = render_args.window.display_texture

    w = render_args.width
    h = render_args.height

    # Process user input and interface
    editor.update_camera(w, h, render_args.delta_time, render_args.window)

    cmd = gpu.CommandList()

    Utility.ClearTarget(
        cmd,
        [1.0, 1.0, 1.0, 0.0],
        output_target, w, h
    )

    StrandRasterizer.Go(
        cmd,
        strands,
        output_target,
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
