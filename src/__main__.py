import coalpy.gpu as gpu
import time

from src import Utility
from src import Editor
from src import Debug
from src import StrandFactory
from src import StrandDeviceMemory
from src import Rasterizer
from src import RasterizerBrute
from src import RasterizerBinned

initial_width  = 1280
initial_height = 720

# Allocate a chunk of device memory resources
device_memory = StrandDeviceMemory.StrandDeviceMemory()

# Create a default strand
strands = StrandFactory.build_from_asset("bezier_dev_multiple")

# Layout the initial memory and bind the position data
device_memory.layout(strands.strand_count, strands.strand_particle_count)
device_memory.bind_strand_position_data(strands.particle_positions)

# Create the rasterizer, allocating internal resources.
# rasterizer = RasterizerBrute.RasterizerBrute(initial_width, initial_height)
rasterizer = RasterizerBinned.RasterizerBinned(initial_width, initial_height)

# Create the debugger
debug = Debug.Debug()

editor = Editor.Editor(device_memory, strands)


def on_render(render_args: gpu.RenderArgs):
    output_target = render_args.window.display_texture

    w = render_args.width
    h = render_args.height

    # Process user input and interface
    editor.update_camera(w, h, render_args.delta_time, render_args.window)

    cmd = gpu.CommandList()

    # Clear the color target.
    cmd.begin_marker("ClearColorTarget")
    Utility.clear_target(
        cmd,
        [0.0, 0.0, 0.0, 0.0],
        output_target, w, h
    )
    cmd.end_marker()

    # Create the new frame context.
    context = Rasterizer.Context(
        cmd, w, h,
        editor.camera.view_matrix,
        editor.camera.proj_matrix,
        device_memory,
        editor.strands.strand_count,
        editor.strands.strand_count * (editor.strands.strand_particle_count - 1),
        editor.strands.strand_particle_count,
        editor.tesselation,
        editor.tesselation_sample_count,
        output_target
    )

    # Invoke the hair strand rasterizer.
    rasterizer.go(context)

    # Crunch some numbers about the rasterizer for this frame.
    stats = debug.compute_stats(
        cmd,
        rasterizer,
        context
    )

    # Debug draw bin counts
    if editor.debug_bin_overlay > 0:
        debug.draw_bin_counts(
            cmd,
            rasterizer,
            context,
            editor.debug_bin_overlay
        )

    editor.render(stats, render_args.imgui)

    # Schedule the work.
    gpu.schedule(cmd)


# Invoke the window creation and register our render loop.
window = gpu.Window("StrandRasterizer", initial_width, initial_height, on_render)

gpu.run()