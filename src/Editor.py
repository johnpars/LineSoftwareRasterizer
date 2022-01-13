# Editor adapted from Kleber Garcia's GRR (GPU Rasterizer and Renderer)

import coalpy.gpu as g
import numpy as np
import sys
import pathlib

from src import StrandFactory
from src import StrandDeviceMemory
from src import Camera as c
from src import Vector
from src import Utility
from src import Debug


class Editor:

    def __init__(self, deviceMemory: StrandDeviceMemory.StrandDeviceMemory, strands: StrandFactory.Strands):
        self.editor_camera = c.Camera(1920, 1080)
        self.editor_camera.pos = Vector.float3(0, 2, -14)
        self.frame_it = 0

        # Temp: debugging camera default
        self.editor_camera.pos = Vector.float3(7.827458, 1.0573884, 7.414528)
        self.editor_camera.transform.rotation = np.quaternion(0.390986784050001, 0.0223924684704286,
                                                              -0.918854440764099, 0.0483159327309966)
        self.editor_camera.fov = 1.132
        self.editor_camera.transform.update_mats()

        # input state
        self.pressed_r = False
        self.pressed_l = False
        self.pressed_t = False
        self.pressed_b = False
        self.pressed_can_move = False

        # camera settings
        self.speed_camera_movement = 0.1
        self.speed_camera_rotation = 0.1
        self.position_mouse_last = (0, 0)

        # strand generation settings
        self.generation_settings = StrandFactory.Settings()
        self.device_memory = deviceMemory
        self.strands = strands

        # rasterizer settings
        self.debug_bin_overlay = 0.0
        self.tesselation = False
        self.tesselation_sample_count = 12

        # ui panels states
        self.panel_camera  = False
        self.panel_strands = False
        self.panel_stats   = True

    @property
    def camera(self):
        return self.editor_camera

    def update_inputs(self, input_states):
        self.pressed_r = input_states.get_key_state(g.Keys.D)
        self.pressed_l = input_states.get_key_state(g.Keys.A)
        self.pressed_t = input_states.get_key_state(g.Keys.W)
        self.pressed_b = input_states.get_key_state(g.Keys.S)

        prev_mouse = self.pressed_can_move
        self.pressed_can_move = input_states.get_key_state(g.Keys.MouseRight)
        if prev_mouse != self.pressed_can_move:
            m = input_states.get_mouse_position()
            self.position_mouse_last = (m[2], m[3])

    def update_camera(self, w, h, delta_time, input_states):
        self.frame_it = self.frame_it + 1
        self.editor_camera.w = w
        self.editor_camera.h = h
        self.update_inputs(input_states)

        if (self.pressed_can_move):
            cam_t = self.editor_camera.transform
            new_pos = self.editor_camera.pos
            zero = Vector.float3(0, 0, 0)
            new_pos = new_pos - ((cam_t.right * self.speed_camera_movement) if self.pressed_r else zero)
            new_pos = new_pos + ((cam_t.right * self.speed_camera_movement) if self.pressed_l else zero)
            new_pos = new_pos + ((cam_t.front * self.speed_camera_movement) if self.pressed_t else zero)
            new_pos = new_pos - ((cam_t.front * self.speed_camera_movement) if self.pressed_b else zero)
            self.editor_camera.pos = new_pos

            curr_mouse = input_states.get_mouse_position()
            rot_vec = delta_time * self.speed_camera_rotation * Vector.float3(curr_mouse[2] - self.position_mouse_last[0],
                                                                              curr_mouse[3] - self.position_mouse_last[1], 0.0)

            x_axis = self.editor_camera.transform.right
            y_axis = Vector.float3(0, 1, 0)

            prev_q = self.editor_camera.transform.rotation
            qx = Vector.q_from_angle_axis(-np.sign(rot_vec[0]) * (np.abs(rot_vec[0]) ** 1.2), y_axis)
            qy = Vector.q_from_angle_axis(np.sign(rot_vec[1]) * (np.abs(rot_vec[1]) ** 1.2), x_axis)
            self.editor_camera.transform.rotation = qy * (qx * prev_q)
            self.editor_camera.transform.update_mats()
            self.position_mouse_last = (curr_mouse[2], curr_mouse[3])

    def render_main_menu_bar(self, imgui: g.ImguiBuilder):
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Tools"):
                self.panel_camera = True if imgui.menu_item(label="Camera") else self.panel_camera
                self.panel_strands = True if imgui.menu_item(label="Strand Generation") else self.panel_strands
                self.panel_stats = True if imgui.menu_item(label="Rasterizer Stats") else self.panel_stats
                imgui.end_menu()
            imgui.end_main_menu_bar()

    def render_camera_bar(self, imgui: g.ImguiBuilder):
        if not self.panel_camera:
            return

        self.panel_camera = imgui.begin("Camera", self.panel_camera)
        if imgui.collapsing_header("params"):
            self.editor_camera.fov = imgui.slider_float(label="fov", v=self.editor_camera.fov, v_min=0.01 * np.pi,
                                                        v_max=0.7 * np.pi)
            self.editor_camera.near = imgui.slider_float(label="near", v=self.editor_camera.near, v_min=0.001,
                                                         v_max=8.0)
            self.editor_camera.far = imgui.slider_float(label="far", v=self.editor_camera.far, v_min=10.0,
                                                        v_max=90000)

        if imgui.collapsing_header("transform"):
            cam_transform = self.editor_camera.transform
            nx = cam_transform.translation[0]
            ny = cam_transform.translation[1]
            nz = cam_transform.translation[2]
            (nx, ny, nz) = imgui.input_float3(label="pos", v=[nx, ny, nz])
            cam_transform.translation = [nx, ny, nz]
            if imgui.button("reset"):
                cam_transform.translation = [0, 0, 0]
                cam_transform.rotation = Vector.q_from_angle_axis(0, Vector.float3(1, 0, 0))
        imgui.end()

    def render_strand_bar(self, imgui: g.ImguiBuilder):
        if not self.panel_strands:
            return

        self.panel_strands = imgui.begin("Strand Generation", self.panel_strands)

        # TODO: Add more imgui bindings, can't do int slider or toggle, etc.
        self.generation_settings.strand_count = imgui.slider_float(label="Strand Count",
                                                                   v=self.generation_settings.strand_count,
                                                                   v_min=1.0, v_max=2048.0)
        self.generation_settings.strand_particle_count = imgui.slider_float(label="Strand Particle Count",
                                                                            v=self.generation_settings.strand_particle_count,
                                                                            v_min=2.0, v_max=64.0)
        self.generation_settings.strand_length = imgui.slider_float(label="Strand Length",
                                                                    v=self.generation_settings.strand_length,
                                                                    v_min=0.001, v_max=5.0)

        if imgui.button("BUILD"):
            self.rebuild_strands()

        imgui.end()

    def render_stats_bar(self, imgui: g.ImguiBuilder, stats: Debug.Stats):
        if not self.panel_stats:
            return

        self.panel_stats = imgui.begin("Rasterizer Stats", self.panel_stats)

        imgui.text("Total Segments:               " + str(stats.segmentCount))
        imgui.text("Frustum Culled (Pass / Fail): {} / {}".format(stats.segmentCountPassedFrustumCull,
                                                                  stats.segmentCount - stats.segmentCountPassedFrustumCull))

        debug_bin_overlay = imgui.slider_float(" Bin Overlay", self.debug_bin_overlay, 0, 1, "%.2f")
        self.debug_bin_overlay = debug_bin_overlay

        self.tesselation = imgui.checkbox("Tesselation", self.tesselation)

        imgui.end()

    def rebuild_strands(self):
        # Need to manually round this due to lack of integer imgui bindings
        self.generation_settings.strand_count = int(round(self.generation_settings.strand_count))
        self.generation_settings.strand_particle_count = int(round(self.generation_settings.strand_particle_count))

        # Produce a strand group based on the input settings...
        self.strands = StrandFactory.build_procedural(self.generation_settings)
        self.strands.strand_count = self.generation_settings.strand_count
        self.strands.strand_particle_count = self.generation_settings.strand_particle_count

        # ...and inform the device to reformat the buffers
        # (position data will be bound later in a separate compute buffer, to similarly resemble the full data model).
        self.device_memory.layout(self.strands.strand_count, self.strands.strand_particle_count)
        self.device_memory.bind_strand_position_data(self.strands.particle_positions)

    def render(self, stats: Debug.Stats, imgui: g.ImguiBuilder):
        self.render_main_menu_bar(imgui)
        self.render_camera_bar(imgui)
        self.render_strand_bar(imgui)
        self.render_stats_bar(imgui, stats)
