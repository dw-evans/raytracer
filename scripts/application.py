from __future__ import annotations


import pygame
import moderngl

from typing import Protocol
from abc import ABC
import struct

from scripts import functions

from scripts import classes
from scripts.classes import (
    Scene,
    Sphere,
    Triangle,
    Mesh,
    Camera,
    Csys,
    Material,
    HitInfo,
)

import pyrr
from pyrr import (
    Vector3,
    Vector4,
    Matrix33,
    Matrix44,
    Quaternion,
)

import numpy as np
from numpy import sin, cos, tan

import threading

import sys
from typing import Generator

from scripts.scenes import basic_scene


class Application:
    # window params
    DYNAMIC_RENDER_FRAMERATE = 60
    WINDOW_HEIGHT = 540
    ASPECT_RATIO = 16 / 9

    # mouse / keyboard movement camera speeds
    CAMERA_LINEAR_SPEED = 3.0
    CAMERA_ANGULAR_SPEED = 5.0
    CAMERA_SCROLL_SPEED = 20.0
    CAMERA_ROLL_SPEED = 20.0

    # opengl data
    display_program: ProgramABC = None
    programs: dict[str, ProgramABC] = {}
    display_scene: Scene = None
    scenes: dict[str, Scene] = {}

    input_thread: threading.Thread = None

    RUNNING: bool = False

    clock: pygame.time.Clock
    screen: pygame.Surface

    @property
    def WINDOW_WIDTH(self):
        return int(self.WINDOW_HEIGHT * self.ASPECT_RATIO)

    @property
    def cam_linear_speed_adjusted(self):
        return 1 / self.DYNAMIC_RENDER_FRAMERATE * self.CAMERA_LINEAR_SPEED

    @property
    def cam_angular_speed_adjusted(self):
        return 1 / self.DYNAMIC_RENDER_FRAMERATE * self.CAMERA_ANGULAR_SPEED

    @property
    def screen_center(self):
        return (self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT // 2)

    @property
    def cam_scroll_speed_adjusted(self):
        return 1 / self.DYNAMIC_RENDER_FRAMERATE * self.CAMERA_SCROLL_SPEED

    @property
    def cam_roll_speed_adjusted(self):
        return 1 / self.DYNAMIC_RENDER_FRAMERATE * self.CAMERA_ROLL_SPEED

    def __init__(
        self,
    ) -> None:
        pygame.init()

        Application.screen = pygame.display.set_mode(
            (self.WINDOW_WIDTH, self.WINDOW_HEIGHT),
            pygame.OPENGL | pygame.DOUBLEBUF,
        )
        Application.clock = pygame.time.Clock()
        self.display_scene = basic_scene.scene

        self.register_program(
            DefaultShaderProgram(
                file_fragment_shader="shaders/dumb.fs.glsl",
                file_vertex_shader="shaders/dumb.vs.glsl",
                width=self.WINDOW_HEIGHT,
                height=self.WINDOW_HEIGHT,
            ),
        )

        self.RUNNING = True
        self.run()

    def register_program(self, program: ProgramABC):
        if not program in self.programs.keys():
            self.programs[program.name] = program
        else:
            print("Warning, program already regiested")

        # if there is only one program, set it to the display program
        if len(list(self.programs.keys())) == 1:
            self.display_program = self.programs[list(self.programs.keys())[0]]

    def run(self):
        # configure the program and then run it
        self.display_program.configure_program(self.display_scene)
        # try:
        while self.RUNNING:
            self.display_program.handle_interactions(
                self.display_scene.cam,
                self.display_scene,
                True,
                True,
            )
            self.display_program.calculate_frame(self.display_scene)
            pygame.display.flip()
            Application.clock.tick(Application.DYNAMIC_RENDER_FRAMERATE)
        # except Exception as e:
        #     print(f"Exception occurred: {e}")


def animate_csys(
    obj: Csys,
    function: callable,
    dt: float,
    quat0: Quaternion | None = None,
    quat1: Quaternion | None = None,
    pos0: Vector3 | None = None,
    pos1: Vector3 | None = None,
) -> Generator[Csys]:

    raise NotImplementedError
    yield obj


class ProgramABC(ABC):
    name: str

    context: moderngl.Context
    program: moderngl.Program

    vbo: moderngl.Buffer
    vao: moderngl.VertexArray
    fbo: moderngl.Framebuffer | None

    file_fragment_shader: str
    file_vertex_shader: str
    fragment_shader: str
    vertex_shader: str

    width: int
    height: int

    vertices: np.ndarray
    standalone: bool

    def __init__(
        self,
        file_vertex_shader: str,
        file_fragment_shader: str,
        width: int,
        height: int,
        standalone=True,
        require=460,
    ) -> None:
        super().__init__()

        self.vertices = np.array(
            [
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                -1.0,
                -1.0,
                0.0,
                1.0,
                -1.0,
                0.0,
            ],
            dtype="f4",
        )
        self.file_fragment_shader = file_fragment_shader
        self.file_vertex_shader = file_vertex_shader
        self.width = width
        self.height = height
        self.standalone = standalone

        self.context = moderngl.create_context(
            require=require,
            standalone=standalone,
        )

        self.initialise()

    def initialise(self):
        self.reload_shaders()

        self.program = self.context.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=self.fragment_shader,
        )
        self.vbo = self.context.buffer(self.vertices.tobytes())
        self.vao = self.context.vertex_array(
            self.program, [(self.vbo, "3f", "position")]
        )

        if self.standalone == False:
            self.fbo = None
        else:
            self.fbo = self.context.framebuffer(
                color_attachments=[self.context.texture((self.width, self.height))],
            )

    def load_shaders(
        self,
        file_fragment_shader: str,
        file_vertex_shader: str,
    ):
        with open(file_fragment_shader) as f:
            self.fragment_shader = f.read()
        with open(file_vertex_shader) as f:
            self.vertex_shader = f.read()

    def reload_shaders(self):
        return self.load_shaders(
            file_fragment_shader=self.file_fragment_shader,
            file_vertex_shader=self.file_vertex_shader,
        )

    def configure_program(self, scene: Scene):
        raise NotImplementedError

    def calculate_frame(self, scene: Scene):
        raise NotImplementedError

    def handle_interactions(
        self,
        cam: Camera,
        scene: Scene,
        keyboard_enabled: bool,
        mouse_enabled=True,
    ):
        raise NotImplementedError


class DefaultShaderProgram(ProgramABC):
    name = "default"

    def __init__(
        self,
        file_vertex_shader: str,
        file_fragment_shader: str,
        width: int,
        height: int,
        standalone=False,
        require=460,
    ) -> None:
        super().__init__(
            file_vertex_shader,
            file_fragment_shader,
            width,
            height,
            standalone,
            require,
        )

    def configure_program(self, scene: Scene):

        program = self.program
        spheres = scene.spheres
        context = self.context

        program["spheresCount"].write(struct.pack("i", len(spheres)))
        self.sphere_buffer_binding = 1
        program["sphereBuffer"].binding = self.sphere_buffer_binding

        triangles = scene.triangles
        n_triangles = len(triangles)
        program["triCount"].write(struct.pack("i", n_triangles))

        program["meshCount"].write(struct.pack("i", len(scene.meshes)))

        triangles_ssbo = context.buffer(
            functions.iter_to_bytes(
                [t.update_pos_with_mesh2() for t in scene.triangles],
            )
        )
        triangles_ssbo_binding = 9
        triangles_ssbo.bind_to_storage_buffer(binding=triangles_ssbo_binding)

        mesh_buffer = context.buffer(functions.iter_to_bytes(scene.meshes))
        mesh_buffer_binding = 10
        program["meshBuffer"].binding = mesh_buffer_binding
        mesh_buffer.bind_to_uniform_block(mesh_buffer_binding)

        sky_color = Vector3((131, 200, 228), dtype="f4") / 255

        program["skyColor"].write(struct.pack("3f", *sky_color))

        # program["RAYS_PER_PIXEL"].write(struct.pack("i", 4))

    def calculate_frame(self, scene: Scene):

        program = self.program
        context = self.context
        cam = scene.cam
        vao = self.vao
        spheres = scene.spheres

        program["ViewParams"].write(cam.view_params.astype("f4"))
        program["CamLocalToWorldMatrix"].write(cam.local_to_world_matrix.astype("f4"))
        program["CamGlobalPos"].write(cam.csys.pos.astype("f4"))

        time = pygame.time.get_ticks() / np.float32(1000.0)

        cam.csys.pos.x = 0 + 2.0 * sin(time)

        # CAM_DEPTH_OF_FIELD_STRENGTH = 0.000
        # prog1["depthOfFieldStrength"].write(
        #     struct.pack("f", CAM_DEPTH_OF_FIELD_STRENGTH)
        # )
        # CAM_ANTIALIAS_STRENGTH = 0.001
        # prog1["depthOfFieldStrength"].write(struct.pack("f", CAM_ANTIALIAS_STRENGTH))

        sphere_bytes = functions.iter_to_bytes(spheres)
        sphere_buffer = context.buffer(sphere_bytes)
        sphere_buffer.bind_to_uniform_block(self.sphere_buffer_binding)

        # if MODIFICATION_WAITING:
        #     MODIFY_COMMAND()
        #     MODIFICATION_WAITING = False

        vao.render(mode=moderngl.TRIANGLE_STRIP)

        context.gc()

    def handle_interactions(
        self,
        cam: Camera,
        scene: Scene,
        keyboard_enabled: bool,
        mouse_enabled=True,
    ):
        return generic_camera_event_handler(cam, scene, keyboard_enabled, mouse_enabled)


class RayTracerStatic(ProgramABC):
    MAX_RAY_BOUNCES = 4
    RAYS_PER_PIXEL = 12
    HIGHLIGHT_MATERIAL = Material(
        Vector4((0.3, 0.6, 0.8, 1.0), dtype="f4"),
        Vector3((0, 0, 0), dtype="f4"),
        0.0,
        smoothness=0.5,
    )
    KEYBOARD_ENABLED = True
    MOUSE_ENABLED = True

    def __init__(
        self,
        file_vertex_shader: str,
        file_fragment_shader: str,
        width: int,
        height: int,
        standalone=True,
        require=460,
    ) -> None:
        super().__init__(
            file_vertex_shader,
            file_fragment_shader,
            width,
            height,
            standalone,
            require,
        )

    def configure_program(self, scene: Scene):
        prog = self.program
        ctx = self.context
        spheres = scene.spheres

        prog["STATIC_RENDER"].write(struct.pack("i", True))
        prog["MAX_BOUNCES"].write(struct.pack("i", self.MAX_RAY_BOUNCES))
        prog["RAYS_PER_PIXEL"].write(struct.pack("i", self.RAYS_PER_PIXEL))

        texture = ctx.texture((self.width, self.height), 3)
        texture.use(location=1)
        prog["previousFrame"] = 1

        # initialise the uniforms
        prog["screenWidth"].write(struct.pack("i", self.width))
        prog["screenHeight"].write(struct.pack("i", self.height))

        prog["spheresCount"].write(struct.pack("i", len(spheres)))
        sphere_buffer_binding = 1
        prog["sphereBuffer"].binding = sphere_buffer_binding

        triangles = scene.triangles
        n_triangles = len(triangles)
        prog["triCount"].write(struct.pack("i", n_triangles))

        prog["meshCount"].write(struct.pack("i", len(scene.meshes)))

        triangles_ssbo = ctx.buffer(
            functions.iter_to_bytes(
                [t.update_pos_with_mesh2() for t in scene.triangles],
            )
        )
        triangles_ssbo_binding = 9
        triangles_ssbo.bind_to_storage_buffer(binding=triangles_ssbo_binding)
        # program["triBuffer"].binding = triangles_ssbo_binding

        mesh_buffer = ctx.buffer(functions.iter_to_bytes(scene.meshes))
        mesh_buffer_binding = 10
        prog["meshBuffer"].binding = mesh_buffer_binding
        mesh_buffer.bind_to_uniform_block(mesh_buffer_binding)

        sky_color = Vector3((131, 200, 228), dtype="f4") / 255

        prog["skyColor"].write(struct.pack("3f", *sky_color))

        material_buffer = ctx.buffer(functions.iter_to_bytes([self.HIGHLIGHT_MATERIAL]))
        material_buffer_binding = 11
        prog["materialBuffer"].binding = material_buffer_binding
        material_buffer.bind_to_uniform_block(material_buffer_binding)

    def calculate_frame(self):
        return super().calculate_frame()

    def handle_interactions(self):
        return super().handle_interactions()


MMB_PRESSED = False


def generic_camera_event_handler(
    cam: Camera,
    scene: Scene,
    keyboard_enabled: bool,
    mouse_enabled=True,
):
    global MMB_PRESSED

    cam_linear_speed_adjusted = Application.cam_linear_speed_adjusted
    cam_roll_speed_adjusted = Application.cam_roll_speed_adjusted
    cam_scroll_speed_adjusted = Application.cam_scroll_speed_adjusted
    cam_angular_speed_adjusted = Application.cam_angular_speed_adjusted
    mouse_sphere = scene.spheres[-1]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()

        if keyboard_enabled:

            key_state = pygame.key.get_pressed()
            if key_state[pygame.K_w]:
                cam.csys.tzp(1 * cam_linear_speed_adjusted)
            if key_state[pygame.K_s]:
                cam.csys.tzp(-1 * cam_linear_speed_adjusted)
            if key_state[pygame.K_d]:
                cam.csys.txp(1 * cam_linear_speed_adjusted)
            if key_state[pygame.K_a]:
                cam.csys.txp(-1 * cam_linear_speed_adjusted)
            if key_state[pygame.K_q]:
                cam.csys.rzp(-1 * cam_roll_speed_adjusted)
            if key_state[pygame.K_e]:
                cam.csys.rzp(cam_roll_speed_adjusted)
            if key_state[pygame.K_SPACE]:
                cam.csys.tyg(1 * cam_linear_speed_adjusted)
            if key_state[pygame.K_LCTRL]:
                cam.csys.tyg(-1 * cam_linear_speed_adjusted)

        if mouse_enabled:
            mouse_x, mouse_y = pygame.mouse.get_pos()

            mouse_ray = functions.convert_screen_coords_to_camera_ray(
                mouse_x,
                mouse_y,
                Application.screen.get_width(),
                Application.screen.get_height(),
                cam=scene.cam,
            )

            hits = functions.rayScene(mouse_ray, scene)
            hits = [h for h in hits if isinstance(h[1], Triangle)]
            if hits:
                # print(
                #     f"n_hits={len(hits)}, {', '.join((str(type(x[1])) for x in hits))}"
                # )
                mouse_sphere.pos = hits[0][2]

            if event.type == pygame.MOUSEBUTTONDOWN:
                # right mouse button
                if event.button == 1:
                    pass
                    # if hits:
                    #     obj = hits[0][1]
                    #     if isinstance(obj, Triangle):
                    #         SELECTED_MESH_ID = obj.parent.mesh_index
                    #         prog1["selectedMeshId"].write(
                    #             struct.pack("i", SELECTED_MESH_ID)
                    #         )
                    #         # print(f"selectedMeshId = {SELECTED_MESH_ID}")
                    # else:
                    #     SELECTED_MESH_ID = -1
                    #     prog1["selectedMeshId"].write(
                    #         struct.pack("i", SELECTED_MESH_ID)
                    #     )
                # middle mouse button
                elif event.button == 2:
                    # store the current position here to reset it later
                    MOUSE_LAST_POS_X = mouse_x
                    MOUSE_LAST_POS_Y = mouse_y
                    MMB_PRESSED = True

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    pass
                elif event.button == 2:
                    MOUSE_LAST_POS_X = mouse_x
                    MOUSE_LAST_POS_Y = mouse_y
                    MMB_PRESSED = False

            if event.type == pygame.MOUSEWHEEL:
                scroll = event.precise_y

                cam.fov += -1 * scroll * cam_scroll_speed_adjusted

            if MMB_PRESSED:
                # pygame.mouse.set_pos(screen_center)
                # mouse_dx, mouse_dy = pygame.mouse.get_rel()

                mouse_dx = mouse_x - MOUSE_LAST_POS_X
                mouse_dy = mouse_y - MOUSE_LAST_POS_Y

                MOUSE_LAST_POS_X = mouse_x
                MOUSE_LAST_POS_Y = mouse_y

                # print(f"mouse_dx, mouse_dy = {mouse_dx, mouse_dy}")

                dyaw = -1 * mouse_dx * cam_angular_speed_adjusted
                dpitch = -1 * mouse_dy * cam_angular_speed_adjusted

                # print(f"dyaw, dpitch = {dyaw, dpitch}")

                # dyaw = 1
                # dpitch = 1

                cam.csys.ryp(dyaw)
                cam.csys.rxp(dpitch)
