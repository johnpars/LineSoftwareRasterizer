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

    def __init__(self, deviceMemory : StrandDeviceMemory.StrandDeviceMemory, strands : StrandFactory.Strands):
        self.m_active_scene_name = None
        self.m_active_scene = None
        self.m_editor_camera = c.Camera(1920, 1080)
        self.m_editor_camera.pos = Vector.float3(0, 2, -14)
        self.m_frame_it = 0

        # Temp: debugging camera default
        self.m_editor_camera.pos = Vector.float3(7.827458, 1.0573884, 7.414528)
        self.m_editor_camera.transform.rotation = np.quaternion(0.390986784050001, 0.0223924684704286, -0.918854440764099, 0.0483159327309966)
        self.m_editor_camera.fov = 1.132
        self.m_editor_camera.transform.update_mats()

        # input state
        self.m_right_pressed = False
        self.m_left_pressed = False
        self.m_top_pressed = False
        self.m_bottom_pressed = False
        self.m_can_move_pressed = False
        self.m_last_mouse = (0.0, 0.0)

        # camera settings
        self.m_cam_move_speed = 0.1
        self.m_cam_rotation_speed = 0.1
        self.m_last_mouse = (0, 0)

        # strand generation settings
        self.m_generation_settings = StrandFactory.Settings()
        self.m_device_memory = deviceMemory
        self.m_strands = strands

        # ui panels states
        self.m_camera_panel = False
        self.m_strand_panel = False
        self.m_stats_panel = True

    @property
    def camera(self):
        return self.m_editor_camera

    def UpdateInputs(self, input_states):
        self.m_right_pressed = input_states.get_key_state(g.Keys.D)
        self.m_left_pressed = input_states.get_key_state(g.Keys.A)
        self.m_top_pressed = input_states.get_key_state(g.Keys.W)
        self.m_bottom_pressed = input_states.get_key_state(g.Keys.S)

        prev_mouse = self.m_can_move_pressed
        self.m_can_move_pressed = input_states.get_key_state(g.Keys.MouseRight)
        if prev_mouse != self.m_can_move_pressed:
            m = input_states.get_mouse_position()
            self.m_last_mouse = (m[2], m[3])

    def UpdateCamera(self, w, h, delta_time, input_states):
        self.m_frame_it = self.m_frame_it + 1
        self.m_editor_camera.w = w
        self.m_editor_camera.h = h
        self.UpdateInputs(input_states)

        if (self.m_can_move_pressed):
            cam_t = self.m_editor_camera.transform
            new_pos = self.m_editor_camera.pos
            zero = Vector.float3(0, 0, 0)
            new_pos = new_pos - ((cam_t.right * self.m_cam_move_speed) if self.m_right_pressed else zero)
            new_pos = new_pos + ((cam_t.right * self.m_cam_move_speed) if self.m_left_pressed else zero)
            new_pos = new_pos + ((cam_t.front * self.m_cam_move_speed) if self.m_top_pressed else zero)
            new_pos = new_pos - ((cam_t.front * self.m_cam_move_speed) if self.m_bottom_pressed else zero)
            self.m_editor_camera.pos = new_pos

            curr_mouse = input_states.get_mouse_position()
            rot_vec = delta_time * self.m_cam_rotation_speed * Vector.float3(curr_mouse[2] - self.m_last_mouse[0],
                                                                          curr_mouse[3] - self.m_last_mouse[1], 0.0)

            x_axis = self.m_editor_camera.transform.right
            y_axis = Vector.float3(0, 1, 0)

            prev_q = self.m_editor_camera.transform.rotation
            qx = Vector.q_from_angle_axis(-np.sign(rot_vec[0]) * (np.abs(rot_vec[0]) ** 1.2), y_axis)
            qy = Vector.q_from_angle_axis(np.sign(rot_vec[1]) * (np.abs(rot_vec[1]) ** 1.2), x_axis)
            self.m_editor_camera.transform.rotation = qy * (qx * prev_q)
            self.m_editor_camera.transform.update_mats()
            self.m_last_mouse = (curr_mouse[2], curr_mouse[3])

    def RenderMainMenuBar(self, imgui: g.ImguiBuilder):
        if (imgui.begin_main_menu_bar()):
            if (imgui.begin_menu("Tools")):
                self.m_camera_panel = True if imgui.menu_item(label="Camera") else self.m_camera_panel
                self.m_strand_panel = True if imgui.menu_item(label="Strand Generation") else self.m_strand_panel
                self.m_stats_panel = True if imgui.menu_item(label="Rasterizer Stats") else self.m_stats_panel
                imgui.end_menu()
            imgui.end_main_menu_bar()

    def RenderCameraBar(self, imgui: g.ImguiBuilder):
        if not self.m_camera_panel:
            return

        self.m_camera_panel = imgui.begin("Camera", self.m_camera_panel)
        if (imgui.collapsing_header("params")):
            self.m_editor_camera.fov = imgui.slider_float(label="fov", v=self.m_editor_camera.fov, v_min=0.01 * np.pi,
                                                          v_max=0.7 * np.pi)
            self.m_editor_camera.near = imgui.slider_float(label="near", v=self.m_editor_camera.near, v_min=0.001,
                                                           v_max=8.0)
            self.m_editor_camera.far = imgui.slider_float(label="far", v=self.m_editor_camera.far, v_min=10.0,
                                                          v_max=90000)

        if (imgui.collapsing_header("transform")):
            cam_transform = self.m_editor_camera.transform
            nx = cam_transform.translation[0]
            ny = cam_transform.translation[1]
            nz = cam_transform.translation[2]
            (nx, ny, nz) = imgui.input_float3(label="pos", v=[nx, ny, nz])
            cam_transform.translation = [nx, ny, nz]
            if (imgui.button("reset")):
                cam_transform.translation = [0, 0, 0]
                cam_transform.rotation = Vector.q_from_angle_axis(0, Vector.float3(1, 0, 0))
        imgui.end()

    def RenderStrandBar(self, imgui: g.ImguiBuilder):
        if not self.m_strand_panel:
            return

        self.m_strand_panel = imgui.begin("Strand Generation", self.m_strand_panel)

        # TODO: Add more imgui bindings, can't do int slider or toggle, etc.
        self.m_generation_settings.strandCount         = imgui.slider_float(label="Strand Count", v=self.m_generation_settings.strandCount, v_min=1.0, v_max=2048.0)
        self.m_generation_settings.strandParticleCount = imgui.slider_float(label="Strand Particle Count", v=self.m_generation_settings.strandParticleCount, v_min=2.0, v_max=64.0)
        self.m_generation_settings.strandLength        = imgui.slider_float(label="Strand Length", v=self.m_generation_settings.strandLength, v_min=0.001, v_max=5.0)

        if imgui.button("BUILD"):
            self.RebuildStrands()


        imgui.end()


    def RenderStatsBar(self, imgui: g.ImguiBuilder, stats: Debug.Stats):
        if not self.m_stats_panel:
            return

        self.m_stats_panel = imgui.begin("Rasterizer Stats", self.m_stats_panel)

        imgui.text("Total Segments:               " + str(stats.segmentCount))
        imgui.text("Frustum Culled (Pass / Fail): {} / {}".format(stats.segmentCountPassedFrustumCull, stats.segmentCount - stats.segmentCountPassedFrustumCull))

        imgui.end()


    def RebuildStrands(self):
        # Need to manually round this due to lack of integer imgui bindings
        self.m_generation_settings.strandCount = int(round(self.m_generation_settings.strandCount))
        self.m_generation_settings.strandParticleCount = int(round(self.m_generation_settings.strandParticleCount))

        # Produce a strand group based on the input settings...
        self.m_strands = StrandFactory.BuildProcedural(self.m_generation_settings)
        self.m_strands.strandCount = self.m_generation_settings.strandCount
        self.m_strands.strandParticleCount = self.m_generation_settings.strandParticleCount

        # ...and inform the device to reformat the buffers
        # (position data will be bound later in a separate compute buffer, to similarly resemble the full data model).
        self.m_device_memory.Layout(self.m_strands.strandCount, self.m_strands.strandParticleCount)
        self.m_device_memory.BindStrandPositionData(self.m_strands.particlePositions)


    def Render(self, stats: Debug.Stats, imgui: g.ImguiBuilder):
        self.RenderMainMenuBar(imgui)
        self.RenderCameraBar(imgui)
        self.RenderStrandBar(imgui)
        self.RenderStatsBar(imgui, stats)

