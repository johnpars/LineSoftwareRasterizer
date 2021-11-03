import math

import coalpy.gpu as gpu

# Configure the graphics device.
gpu.set_current_adapter(0)


def BuildUserInterface(imgui):
    imgui.begin("Settings")
    imgui.end()


def OnRender(render_args: gpu.RenderArgs):
    # Prepare the UI for this tick.
    BuildUserInterface(render_args.imgui)

    cmd = gpu.CommandList()

    dim_x = int(math.ceil(render_args.width / 8))
    dim_y = int(math.ceil(render_args.height / 8))

    # Plan some work.
    cmd.dispatch(
        x=dim_x, y=dim_y, z=1,
        constants=[
            float(render_args.width), float(render_args.height), float(0.001 * render_args.render_time), 0.0
        ],
        shader=Shader,
        outputs=render_args.window.display_texture
    )

    # Schedule the work.
    gpu.schedule(
        [cmd]
    )


Shader = gpu.Shader(file="src/shaders/HelloWorld.hlsl", main_function="Main")
Window = gpu.Window("StrandRasterizer", 1920, 1080, OnRender)
Output = gpu.OutResourceTable("Output", [Window.display_texture])

if __name__ == '__main__':
    gpu.run()
