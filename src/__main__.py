import coalpy.gpu as gpu

from src import Utility
from src import Editor
from src import Debug
from src import StrandFactory
from src import StrandDeviceMemory
from src import StrandRasterizer

initial_width  = 1280
initial_height = 720

# Allocate a chunk of device memory resources
deviceMemory = StrandDeviceMemory.StrandDeviceMemory()

# Create a default strand
strands = StrandFactory.build_from_asset("cube_hair")

# Layout the initial memory and bind the position data
deviceMemory.layout(strands.strand_count, strands.strand_particle_count)
deviceMemory.bind_strand_position_data(strands.particle_positions)

# Create the rasterizer, allocating internal resources.
rasterizer = StrandRasterizer.StrandRasterizer(initial_width, initial_height)

# Create the debugger
debug = Debug.Debug()

editor = Editor.Editor(deviceMemory, strands)


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
    context = StrandRasterizer.Context(
        cmd, w, h,
        editor.camera.view_matrix,
        editor.camera.proj_matrix,
        deviceMemory,
        editor.strands.strand_count,
        editor.strands.strand_count * (editor.strands.strand_particle_count - 1),
        editor.strands.strand_particle_count,
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

    editor.render(stats, render_args.imgui)

    # Schedule the work.
    gpu.schedule(cmd)


# Invoke the window creation and register our render loop.
window = gpu.Window("StrandRasterizer", initial_width, initial_height, on_render)

gpu.run()
