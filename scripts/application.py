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

    def __init__(
        self,
    ) -> None:
        # window params
        self.DYNAMIC_RENDER_FRAMERATE = 60
        self.WINDOW_HEIGHT = 540
        self.ASPECT_RATIO = 16 / 9

        # mouse / keyboard movement camera speeds
        self.CAMERA_LINEAR_SPEED = 3.0
        self.CAMERA_ANGULAR_SPEED = 5.0
        self.CAMERA_SCROLL_SPEED = 20.0
        self.CAMERA_ROLL_SPEED = 20.0

        # opengl data
        self.display_program: ProgramABC = None
        self.programs: dict[str, ProgramABC] = {}
        self.display_scene: Scene = None
        self.scenes: dict[str, Scene] = {}

        self.input_thread: threading.Thread = None

        self.running: bool = False

        self.clock: pygame.time.Clock = None
        self.screen: pygame.Surface = None

        self.MOUSE_LAST_POS: list[float] = [0, 0]
        self.MMB_PRESSED: bool = False

        pygame.init()

        self.screen = pygame.display.set_mode(
            (self.WINDOW_WIDTH, self.WINDOW_HEIGHT),
            pygame.OPENGL | pygame.DOUBLEBUF,
        )
        self.clock = pygame.time.Clock()
        # self.display_scene = basic_scene.scene
        # self.display_scene = basic_scene.scene2
        # self.display_scene = basic_scene.scene3
        self.display_scene = basic_scene.scene4

        self.display_scene.cam.fov = 45

        self.register_program(
            # DefaultShaderProgram(
            #     app=self,
            #     file_fragment_shader="shaders/dumb.fs.glsl",
            #     file_vertex_shader="shaders/dumb.vs.glsl",
            #     width=self.WINDOW_WIDTH,
            #     height=self.WINDOW_HEIGHT,
            # ),
            RayTracerDynamic(
                app=self,
                file_fragment_shader="shaders/raytracer.fs.glsl",
                file_vertex_shader="shaders/raytracer.vs.glsl",
                width=self.WINDOW_WIDTH,
                height=self.WINDOW_HEIGHT,
            ),
        )

    def start(self):
        self.running = True
        self.run()

    def end(self):
        self.running = False

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
        while self.running:
            self.display_program.handle_interactions(
                self,
                self.display_scene.cam,
                self.display_scene,
                keyboard_enabled=True,
                mouse_enabled=True,
            )
            self.display_program.calculate_frame(self.display_scene)
            pygame.display.flip()
            self.clock.tick(self.DYNAMIC_RENDER_FRAMERATE)
        # except Exception as e:
        #     print(f"Exception occurred: {e}")


def animate_csys(
    obj: Csys,
    f_t: callable,
    dt: float,
    quat0: Quaternion | None = None,
    quat1: Quaternion | None = None,
    pos0: Vector3 | None = None,
    pos1: Vector3 | None = None,
) -> Generator[Csys]:
    pass


class ProgramABC(ABC):
    """Abstract base for a program"""

    app: Application
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
        app: Application,
        file_vertex_shader: str,
        file_fragment_shader: str,
        width: int,
        height: int,
        standalone=True,
        require=460,
    ) -> None:
        super().__init__()
        self.app = app
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
                color_attachments=[
                    self.context.texture((self.width, self.height), components=4)
                ],
            )

    def load_shaders(
        self,
        file_fragment_shader: str,
        file_vertex_shader: str,
    ) -> None:
        with open(file_fragment_shader) as f:
            self.fragment_shader = f.read()
        with open(file_vertex_shader) as f:
            self.vertex_shader = f.read()

    def reload_shaders(self):
        return self.load_shaders(
            file_fragment_shader=self.file_fragment_shader,
            file_vertex_shader=self.file_vertex_shader,
        )

    def configure_program(self, scene: Scene) -> None:
        raise NotImplementedError

    def calculate_frame(self, scene: Scene) -> None:
        raise NotImplementedError

    def handle_interactions(
        self,
        screen: pygame.Surface,
        cam: Camera,
        scene: Scene,
        keyboard_enabled: bool,
        mouse_enabled: bool,
    ):
        raise NotImplementedError


class DefaultShaderProgram(ProgramABC):
    name = "default"

    def __init__(
        self,
        app: Application,
        file_vertex_shader: str,
        file_fragment_shader: str,
        width: int,
        height: int,
        standalone=False,
        require=460,
    ) -> None:
        super().__init__(
            app,
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

        program["selectedMeshId"].write(struct.pack("i", -1))

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

        # cam.csys.pos.x = 0 + 2.0 * sin(time)

        # CAM_DEPTH_OF_FIELD_STRENGTH = 0.000
        # program["depthOfFieldStrength"].write(
        #     struct.pack("f", CAM_DEPTH_OF_FIELD_STRENGTH)
        # )
        # CAM_ANTIALIAS_STRENGTH = 0.001
        # program["depthOfFieldStrength"].write(struct.pack("f", CAM_ANTIALIAS_STRENGTH))

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
        screen: pygame.Surface,
        cam: Camera,
        scene: Scene,
        keyboard_enabled: bool,
        mouse_enabled: bool,
    ):
        return generic_camera_event_handler(
            app=self.app,
            cam=cam,
            scene=scene,
            keyboard_enabled=keyboard_enabled,
            mouse_enabled=mouse_enabled,
        )


class RayTracerDynamic(ProgramABC):
    name = "raytracer"

    def __init__(
        self,
        app: Application,
        file_vertex_shader: str,
        file_fragment_shader: str,
        width: int,
        height: int,
        standalone=False,
        require=460,
    ) -> None:
        super().__init__(
            app,
            file_vertex_shader,
            file_fragment_shader,
            width,
            height,
            standalone,
            require,
        )
        self.MAX_RAY_BOUNCES = 2
        self.RAYS_PER_PIXEL = 16

        self.sphere_buffer_binding = 1
        self.is_scene_static = True

    def configure_program(self, scene: Scene):

        program = self.program
        spheres = scene.spheres
        context = self.context

        self.texture = self.context.texture((self.width, self.height), 3)
        self.texture.use(location=1)
        self.program["previousFrame"] = 1

        self.frame_counter = 0
        self.cycle_counter = 0
        self.shader_rng_counter = 0

        program["STATIC_RENDER"].write(struct.pack("i", True))
        program["MAX_BOUNCES"].write(struct.pack("i", self.MAX_RAY_BOUNCES))
        program["RAYS_PER_PIXEL"].write(struct.pack("i", self.RAYS_PER_PIXEL))
        program["RAYS_PER_PIXEL"].write(struct.pack("i", 1))

        texture = context.texture((self.width, self.height), 3)
        texture.use(location=1)
        program["previousFrame"] = 1

        # initialise the uniforms
        program["screenWidth"].write(struct.pack("i", self.width))
        program["screenHeight"].write(struct.pack("i", self.height))

        program["spheresCount"].write(struct.pack("i", len(spheres)))
        sphere_buffer_binding = 1
        program["sphereBuffer"].binding = sphere_buffer_binding

        triangles = scene.triangles
        n_triangles = len(triangles)
        program["triCount"].write(struct.pack("i", n_triangles))

        program["meshCount"].write(struct.pack("i", len(scene.meshes)))

        self.triangles_ssbo = context.buffer(
            functions.iter_to_bytes(
                [t.update_pos_with_mesh2() for t in scene.triangles],
            )
        )
        triangles_ssbo_binding = 9
        self.triangles_ssbo.bind_to_storage_buffer(binding=triangles_ssbo_binding)

        mesh_buffer = context.buffer(functions.iter_to_bytes(scene.meshes))
        mesh_buffer_binding = 10
        program["meshBuffer"].binding = mesh_buffer_binding
        mesh_buffer.bind_to_uniform_block(mesh_buffer_binding)

        # sky_color = Vector3((131, 200, 228), dtype="f4") / 255
        # program["skyColor"].write(struct.pack("3f", *sky_color))

        material_buffer = context.buffer(basic_scene.atmosphere_material.tobytes())
        material_buffer_binding = 11
        program["materialBuffer"].binding = material_buffer_binding
        material_buffer.bind_to_uniform_block(material_buffer_binding)

        program

    def calculate_frame(self, scene: Scene):

        vao = self.vao
        program = self.program
        context = self.context
        cam = scene.cam

        spheres = scene.spheres

        self.program["MAX_BOUNCES"].write(struct.pack("i", self.MAX_RAY_BOUNCES))
        print(self.MAX_RAY_BOUNCES)

        # self.is_scene_static = False
        # time = pygame.time.get_ticks() / np.float32(1000.0)
        # scene.meshes[0].csys.pos.z = 0.0 + 5.0 * sin(time / 2)
        # scene.spheres[-2].pos.x = 2 + 1.0 * sin(time)

        self.triangles_ssbo.write(
            functions.iter_to_bytes(
                [t.update_pos_with_mesh2() for t in scene.triangles],
            )
        )

        program["frameNumber"].write(struct.pack("I", self.cycle_counter))
        CAM_DEPTH_OF_FIELD_STRENGTH = 0.000
        program["depthOfFieldStrength"].write(
            struct.pack("f", CAM_DEPTH_OF_FIELD_STRENGTH),
        )
        CAM_ANTIALIAS_STRENGTH = 0.000001
        program["depthOfFieldStrength"].write(
            struct.pack("f", CAM_ANTIALIAS_STRENGTH),
        )

        program["ViewParams"].write(cam.view_params.astype("f4"))
        program["CamLocalToWorldMatrix"].write(cam.local_to_world_matrix.astype("f4"))
        program["CamGlobalPos"].write(cam.csys.pos.astype("f4"))

        sphere_bytes = functions.iter_to_bytes(spheres)
        sphere_buffer = context.buffer(sphere_bytes)
        sphere_buffer.bind_to_uniform_block(self.sphere_buffer_binding)

        self.texture.use(location=1)

        vao.render(mode=moderngl.TRIANGLE_STRIP)

        render_data = context.screen.read(components=3, dtype="f1")
        self.texture.write(render_data)

        if self.is_scene_static:
            self.cycle_counter += 1
        else:
            self.is_scene_static = True
            self.cycle_counter = 0

        self.shader_rng_counter += 1

        self.context.gc()

    def handle_interactions(
        self,
        app: Application,
        cam: Camera,
        scene: Scene,
        keyboard_enabled: bool,
        mouse_enabled: bool,
    ):
        """
        If there is any movement of the camera, set flag to reset the cycles
        """

        cam_linear_speed_adjusted = app.cam_linear_speed_adjusted
        cam_roll_speed_adjusted = app.cam_roll_speed_adjusted
        cam_scroll_speed_adjusted = app.cam_scroll_speed_adjusted
        cam_angular_speed_adjusted = app.cam_angular_speed_adjusted

        # mouse_sphere = scene.spheres[-1]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

            if keyboard_enabled:

                key_state = pygame.key.get_pressed()
                if key_state[pygame.K_w]:
                    cam.csys.tzp(1 * cam_linear_speed_adjusted)
                    self.is_scene_static = False
                if key_state[pygame.K_s]:
                    cam.csys.tzp(-1 * cam_linear_speed_adjusted)
                    self.is_scene_static = False
                if key_state[pygame.K_d]:
                    cam.csys.txp(1 * cam_linear_speed_adjusted)
                    self.is_scene_static = False
                if key_state[pygame.K_a]:
                    cam.csys.txp(-1 * cam_linear_speed_adjusted)
                    self.is_scene_static = False
                if key_state[pygame.K_q]:
                    cam.csys.rzp(-1 * cam_roll_speed_adjusted)
                    self.is_scene_static = False
                if key_state[pygame.K_e]:
                    cam.csys.rzp(cam_roll_speed_adjusted)
                    self.is_scene_static = False
                if key_state[pygame.K_SPACE]:
                    cam.csys.tyg(1 * cam_linear_speed_adjusted)
                    self.is_scene_static = False
                if key_state[pygame.K_LCTRL]:
                    cam.csys.tyg(-1 * cam_linear_speed_adjusted)
                    self.is_scene_static = False

                if key_state[pygame.K_0]:
                    self.MAX_RAY_BOUNCES = 0
                    self.is_scene_static = False
                if key_state[pygame.K_1]:
                    self.MAX_RAY_BOUNCES = 1
                    self.is_scene_static = False
                if key_state[pygame.K_2]:
                    self.MAX_RAY_BOUNCES = 2
                    self.is_scene_static = False
                if key_state[pygame.K_3]:
                    self.MAX_RAY_BOUNCES = 3
                    self.is_scene_static = False
                if key_state[pygame.K_4]:
                    self.MAX_RAY_BOUNCES = 4
                    self.is_scene_static = False
                if key_state[pygame.K_5]:
                    self.MAX_RAY_BOUNCES = 5
                    self.is_scene_static = False
                if key_state[pygame.K_6]:
                    self.MAX_RAY_BOUNCES = 6
                    self.is_scene_static = False
                if key_state[pygame.K_7]:
                    self.MAX_RAY_BOUNCES = 7
                    self.is_scene_static = False
                if key_state[pygame.K_8]:
                    self.MAX_RAY_BOUNCES = 8
                    self.is_scene_static = False

            if mouse_enabled:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                mouse_ray = functions.convert_screen_coords_to_camera_ray(
                    mouse_x,
                    mouse_y,
                    app.screen.get_width(),
                    app.screen.get_height(),
                    cam=scene.cam,
                )
                hits = functions.rayScene(mouse_ray, scene)
                hits = [h for h in hits if isinstance(h[1], Triangle | Sphere)]
                if hits:
                    # mouse_sphere.pos = hits[0][2]
                    pass

                if event.type == pygame.MOUSEBUTTONDOWN:
                    # middle mouse button
                    if event.button == 2:
                        # store the current position here to reset it later
                        app.MOUSE_LAST_POS[0] = mouse_x
                        app.MOUSE_LAST_POS[1] = mouse_y
                        # set the flag
                        app.MMB_PRESSED = True

                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        pass
                    elif event.button == 2:
                        app.MOUSE_LAST_POS[0] = mouse_x
                        app.MOUSE_LAST_POS[1] = mouse_y
                        app.MMB_PRESSED = False

                if event.type == pygame.MOUSEWHEEL:
                    scroll = event.precise_y

                    scene.cam.fov += -1 * scroll * cam_scroll_speed_adjusted
                    self.is_scene_static = False

                if app.MMB_PRESSED:

                    mouse_dx = mouse_x - app.MOUSE_LAST_POS[0]
                    mouse_dy = mouse_y - app.MOUSE_LAST_POS[1]

                    app.MOUSE_LAST_POS[0] = mouse_x
                    app.MOUSE_LAST_POS[1] = mouse_y

                    print(app.MOUSE_LAST_POS)

                    dyaw = -1 * mouse_dx * cam_angular_speed_adjusted
                    dpitch = -1 * mouse_dy * cam_angular_speed_adjusted

                    cam.csys.ryg(dyaw)
                    cam.csys.rxp(dpitch)
                    self.is_scene_static = False

                    pass


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


def generic_camera_event_handler(
    app: Application,
    cam: Camera,
    scene: Scene,
    keyboard_enabled: bool,
    mouse_enabled: bool,
):
    """Basic mouse/keyboard handler function for the application"""

    cam_linear_speed_adjusted = app.cam_linear_speed_adjusted
    cam_roll_speed_adjusted = app.cam_roll_speed_adjusted
    cam_scroll_speed_adjusted = app.cam_scroll_speed_adjusted
    cam_angular_speed_adjusted = app.cam_angular_speed_adjusted

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
                app.screen.get_width(),
                app.screen.get_height(),
                cam=scene.cam,
            )
            hits = functions.rayScene(mouse_ray, scene)
            hits = [
                h
                for h in hits
                if isinstance(h[1], Triangle | Sphere) and not h[1] == mouse_sphere
            ]
            if hits:
                mouse_sphere.pos = hits[0][2]

            if event.type == pygame.MOUSEBUTTONDOWN:
                # middle mouse button
                if event.button == 2:
                    # store the current position here to reset it later
                    app.MOUSE_LAST_POS[0] = mouse_x
                    app.MOUSE_LAST_POS[1] = mouse_y
                    # set the flag
                    app.MMB_PRESSED = True
                    print("mmb pressed")

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    pass
                elif event.button == 2:
                    app.MOUSE_LAST_POS[0] = mouse_x
                    app.MOUSE_LAST_POS[1] = mouse_y
                    app.MMB_PRESSED = False

            if event.type == pygame.MOUSEWHEEL:
                scroll = event.precise_y

                scene.cam.fov += -1 * scroll * cam_scroll_speed_adjusted

            if app.MMB_PRESSED:

                mouse_dx = mouse_x - app.MOUSE_LAST_POS[0]
                mouse_dy = mouse_y - app.MOUSE_LAST_POS[1]

                app.MOUSE_LAST_POS[0] = mouse_x
                app.MOUSE_LAST_POS[1] = mouse_y

                print(app.MOUSE_LAST_POS)

                dyaw = -1 * mouse_dx * cam_angular_speed_adjusted
                dpitch = -1 * mouse_dy * cam_angular_speed_adjusted

                cam.csys.ryg(dyaw)
                cam.csys.rxp(dpitch)
                pass
