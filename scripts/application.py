from __future__ import annotations


import pygame
import moderngl

from typing import Protocol
from abc import ABC
import struct

from scripts import functions

from . import classes
from .classes import (
    Scene,
    Sphere,
    Mesh,
    Camera,
    Material,
    HitInfo,
)

import scripts.numba_utils.classes
from scripts.numba_utils.classes import Triangle, Csys
from scripts.numba_utils.functions import timer

import pyrr
from pyrr import (
    Vector3,
    Vector4,
    Matrix33,
    Matrix44,
    Quaternion,
)


import numpy as np

import threading
import time

import sys
from typing import Generator

# from scripts.scenes import basic_scene
# from scripts.scenes import animated_scene
# from scripts.scenes import numba_test_scene
from scripts.scenes import final_scene_numba

SCENE = final_scene_numba

def reload_scene():
    pass

from pathlib import Path
import datetime

from functools import partial
import threading

import importlib


from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# MAX_TRIANGLES_TO_LOAD = 1_000


def check_for_errors(ctx: moderngl.Context):
    error = ctx.error
    if error:
        print(f"OpenGL Error: {error}")


class Application:

    def __init__(
        self,
    ) -> None:
        # window params

        factor = 1
        self.DYNAMIC_RENDER_FRAMERATE = 24


        self.RENDER_HEIGHT = 1080 // factor
        self.WINDOW_HEIGHT = max(540, self.RENDER_HEIGHT)
        # self.WINDOW_HEIGHT = 1440 // 8
        self.ASPECT_RATIO = 16 / 9

        self.MAX_CYCLES = 2048 * 2048

        # self.CHUNKSX = 32 // factor
        # self.CHUNKSY = 32 // factor
        self.CHUNKSX = 1
        self.CHUNKSY = 1

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
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_FLAGS, pygame.GL_CONTEXT_DEBUG_FLAG
        )

        self.clock = pygame.time.Clock()
        self.display_scene = SCENE.scene

        # self.display_scene.validate_mesh_indices()

        # self.display_scene.cam.fov = 30

        self.register_program(
            DefaultShaderProgram(
                app=self,
                file_fragment_shader="shaders/dumb.fs.glsl",
                file_vertex_shader="shaders/dumb.vs.glsl",
                width=self.RENDER_WIDTH,
                height=self.RENDER_HEIGHT,
            ),
        )
        # Something can generate a memory leak here
        # It literally happens if I change anything in the main function
        # wtf...
        self.register_program(
            RayTracerDynamic(
                app=self,
                file_fragment_shader="shaders/raytracer.fs.glsl",
                file_vertex_shader="shaders/raytracer.vs.glsl",
                width=self.RENDER_WIDTH,
                height=self.RENDER_HEIGHT,
            ),
        )
        # self.register_program(
        #     RayTracerStatic(
        #         app=self,
        #         file_fragment_shader="shaders/raytracer.fs.glsl",
        #         file_vertex_shader="shaders/raytracer.vs.glsl",
        #         width=self.WINDOW_WIDTH,
        #         height=self.WINDOW_HEIGHT,
        #     ),
        # )

        self.is_waiting_to_toggle = False
        self.is_animating = False
        self.animation_clock = 0
        self.dt = 1 / self.DYNAMIC_RENDER_FRAMERATE
        self.reset_anim_key_pressed = False
        self.pause_play_anim_key_pressed = False
        self.frame_counter = 0
        self.export_directory = (
            Path()
            / "renders/application"
            / f"{datetime.datetime.now().strftime("%Y.%m.%d_%H%M%S")}"
        )
        self.export_directory.mkdir(parents=True)

        self.watchdog: threading.Thread = threading.Thread(
            target=self.run_file_watchdog
        )
        self.watchdog_command: callable = lambda: None
        self.watchdog_command_waiting = False
        self.watchdog_is_running = False

        # self.display_program = self.programs["raytracer_static"]
        self.display_program = self.programs["raytracer"]

        self.is_requesting_exit = threading.Event()

    def start(self):
        self.running = True
        self.run()

    def end(self):
        self.running = False

    @property
    def RENDER_WIDTH(self):
        return int(self.RENDER_HEIGHT * self.ASPECT_RATIO)
    
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
        return (self.RENDER_WIDTH // 2, self.RENDER_HEIGHT // 2)

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
        else:
            self.display_program = self.programs["default"]

        pass

    def toggle_renderer(self):
        """Function to toggle between the raytracer and the dumb one"""
        if self.display_program.name == "default":
            self.display_program = self.programs["raytracer"]
        elif self.display_program.name == "raytracer":
            self.display_program = self.programs["default"]

    def run(self):
        print("Running application loop...")
        # configure the program and then run it
        if not self.watchdog_is_running:
            self.watchdog.start()
            self.watchdog_is_running = True

        print("Reconfiguring display program...")
        self.display_program.configure_program(self.display_scene)
        print("Display program reconfigured...")

        self.is_requesting_exit.clear()

        while self.running:
            if self.watchdog_command_waiting:
                self.watchdog_command_waiting = False
                self.watchdog_command.__call__()
                pass

            options = {
                "app": self,
                "cam": self.display_scene.cam,
                "scene": self.display_scene,
                "keyboard_enabled": True,
                "mouse_enabled": True,
            }

            program_event_handler = partial(
                self.display_program.handle_event,
                **options,
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    # TAB is used to togggle which renderer to use. Not sure how it knows which
                    # one to be writing to the screen
                    if event.key == pygame.K_TAB:
                        if not self.is_waiting_to_toggle:
                            self.is_waiting_to_toggle = True
                            self.toggle_renderer()
                            self.run()

                    # H will be used to return to time zero
                    elif event.key == pygame.K_h:
                        if not self.reset_anim_key_pressed:
                            self.reset_anim_key_pressed = True
                            self.is_animating = False
                            self.animation_clock = 0
                    # J will be used to play/pause
                    elif event.key == pygame.K_j:
                        if not self.pause_play_anim_key_pressed:
                            self.pause_play_anim_key_pressed = True
                            if not self.is_animating:
                                self.is_animating = True
                            else:
                                self.is_animating = False

                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_TAB:
                        self.is_waiting_to_toggle = False

                    elif event.key == pygame.K_h:
                        self.reset_anim_key_pressed = False

                    elif event.key == pygame.K_j:
                        self.pause_play_anim_key_pressed = False

                program_event_handler(event)

            self.display_program.calculate_frame(self.display_scene)
            # pygame.display.flip()
            self.clock.tick(self.DYNAMIC_RENDER_FRAMERATE)

        self.watchdog.join()


    # def save_frame(self):
    #     buffer = self.display_program.context.screen.read(components=3, dtype="f1")
    #     size = (self.display_program.width, self.display_program.height)
    #     img = functions.buffer_to_image(buffer, size)
    #     img.save(self.export_directory / f"{self.frame_counter:05}.png")


    def animate_next(self):
        if self.is_animating:
            print(self.animation_clock)
            self.display_scene.animate(time=self.animation_clock)
            # self.display_program.configure_program(self.display_scene)

            self.animation_clock = self.animation_clock + self.dt
    def animate_frame(self, i):
        if self.is_animating:
            print(f"Frame Number Animation = {i}")
            self.display_scene.animate_frame(frame=i)
            self.display_program.configure_buffers_and_uniforms(self.display_scene)


    class ShaderHandler(FileSystemEventHandler):
        def __init__(self, app: Application):
            self.app = app

        def on_modified(self, event):
            # This method is called when a file is modified
            if not self.app.watchdog_command_waiting and not event.is_directory:
                print(f"File modified: {event.src_path}, reloading shaders")
                self.out_fn_waiting = True
                self.app.is_requesting_exit.set()

                def out_fn():
                    print("File Modification occurred, reloading data...")
                    print("Reloading shaders...")
                    try:
                        self.app.display_program.initialise()
                        print("Shaders reloaded.")
                    except Exception as e:
                        print(f"Error during shader reloading. Check for errors: \n\n{e}\n")
                    print("Reloading scene...")
                    try:
                        importlib.reload(SCENE)
                    except Exception as e:
                        print("Warning, error during import")
                        print(e)
                
                    self.app.display_scene = SCENE.scene
                    print("Scene reloaded")
                    # print("Reconfiguring program...")
                    # self.app.display_program.configure_program(scene=self.app.display_scene)
                    # print("Program reconfigured...")
                    print("Re-running application loop.")
                    self.app.run()
                
                self.app.watchdog_command = out_fn
                self.app.watchdog_command_waiting = True

    def run_file_watchdog(self):
        event_handler = self.ShaderHandler(app=self)
        observer = Observer()
        paths = [
            "shaders",
        ]
        # paths += list(Path("scripts/scenes").glob("*.py"))
        for path in paths:
            observer.schedule(event_handler, path, recursive=False)
            print(f"Started watching {path} for modifications.")


        class Ehandler(FileSystemEventHandler):
            def on_modified(self, event):
                if not Path(event.src_path).resolve().__str__().lower() == SCENE.__file__.lower():
                    return
                # try:
                #     importlib.reload(SCENE)
                # except Exception as e:
                #     print("Warning, error during import")
                #     print(e)
                #     # time.sleep(5.0)
                event_handler.on_modified(event=event)


        observer2 = Observer()
        observer2.schedule(Ehandler(), Path(SCENE.__file__).parent, recursive=False)

        observer.start()
        observer2.start()

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            observer.stop()
            observer2.stop()
        observer.join()
        observer2.join()


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
            debug=True,
        )

        self.initialise()

    def initialise(self):
        self.reload_shaders()

        pass
        # memory leak happens here with the raytracer
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
                    self.context.texture((self.width, self.height), components=3)
                ],
            )

        # self.program.release()

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
        self.load_shaders(
            file_fragment_shader=self.file_fragment_shader,
            file_vertex_shader=self.file_vertex_shader,
        )

    def configure_program(self, scene: Scene) -> None:
        raise NotImplementedError
    
    def configure_buffers_and_uniforms(self, scene:Scene) -> None:
        raise NotImplementedError

    def calculate_frame(self, scene: Scene) -> None:
        raise NotImplementedError

    def handle_event(
        self,
        event,
        app: Application,
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
        # triangles = numba_scripts.classes.ALL_TRIANGLES
        n_triangles = len(triangles)
        program["triCount"].write(struct.pack("i", n_triangles))
        # program["triCount"].write(struct.pack("i", 80000))

        program["meshCount"].write(struct.pack("i", len(scene.meshes)))

        program["selectedMeshId"].write(struct.pack("i", -1))

        print(f"Updating Triangle Positions...")
        # numba_scripts.classes.update_triangles_to_csys(triangles, scene.meshes[0].csys)
        timer(scripts.numba_utils.classes.update_triangles_to_csys)(
            triangles, scene.meshes[0].csys
        )

        print(f"Loading triangles into SSBO...")
        # triangle_data = numba_scripts.classes.triangles_to_array(triangles)
        triangle_data = timer(scripts.numba_utils.classes.triangles_to_array)(triangles)
        # triangle_data = numba_scripts.classes.many_triangles_to_bytes(triangles)
        triangle_bytes = triangle_data.tobytes()

        print(f"Loading triangles complete.")

        self.triangles_ssbo = context.buffer(triangle_bytes)
        triangles_ssbo_binding = 9
        self.triangles_ssbo.bind_to_storage_buffer(binding=triangles_ssbo_binding)

        mesh_buffer = context.buffer(functions.iter_to_bytes(scene.meshes))
        mesh_buffer_binding = 10
        program["meshBuffer"].binding = mesh_buffer_binding
        mesh_buffer.bind_to_uniform_block(mesh_buffer_binding)

        sky_color = Vector3((131, 200, 228), dtype="f4") / 255
        program["skyColor"].write(struct.pack("3f", *sky_color))

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
        if len(sphere_bytes) > 0:
            sphere_buffer = context.buffer(sphere_bytes)
            sphere_buffer.bind_to_uniform_block(self.sphere_buffer_binding)

        # if MODIFICATION_WAITING:
        #     MODIFY_COMMAND()

        vao.render(mode=moderngl.TRIANGLE_STRIP)

        context.gc()

    def handle_event(
        self,
        event,
        app: Application,
        cam: Camera,
        scene: Scene,
        keyboard_enabled: bool,
        mouse_enabled: bool,
    ):
        """Basic mouse/keyboard handler function for the application"""
        # return generic_camera_event_handler(
        #     app=self.app,
        #     cam=cam,
        #     scene=scene,
        #     keyboard_enabled=keyboard_enabled,
        #     mouse_enabled=mouse_enabled,
        # )

        app = self.app

        cam_linear_speed_adjusted = app.cam_linear_speed_adjusted
        cam_roll_speed_adjusted = app.cam_roll_speed_adjusted
        cam_scroll_speed_adjusted = app.cam_scroll_speed_adjusted
        cam_angular_speed_adjusted = app.cam_angular_speed_adjusted

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
            # mouse_ray = functions.convert_screen_coords_to_camera_ray(
            #     mouse_x,
            #     mouse_y,
            #     app.screen.get_width(),
            #     app.screen.get_height(),
            #     cam=scene.cam,
            # )
            # hits = functions.rayScene(mouse_ray, scene)
            # hits = [
            #     h
            #     for h in hits
            #     if isinstance(h[1], Triangle | Sphere) and not h[1] == mouse_sphere
            # ]
            # if hits:
            #     mouse_sphere.pos = hits[0][2]

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

from .scenes.chunker import BVHGraph, BVHParentNode
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
        self.MAX_RAY_BOUNCES = 8
        self.RAYS_PER_PIXEL = 32

        self.sphere_buffer_binding = 1
        self.is_scene_static = True


    def configure_program(self, scene: Scene):

        program = self.program
        spheres = scene.spheres
        context = self.context

        vertices = np.array(
            [
                # x, y, u, v
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype="f4",
        )

        with open("shaders/toScreen.fs.glsl") as f:
            fs = f.read()
        with open("shaders/toScreen.vs.glsl") as f:
            vs = f.read()

        self.screen_prog = self.context.program(
            vertex_shader=vs,
            fragment_shader=fs,
        )

        self.screen_vbo = context.buffer(vertices.tobytes())
        self.screen_vao = context.simple_vertex_array(
            self.screen_prog, self.screen_vbo, "in_pos", "in_uv"
        )

        self.texA = self.context.texture((self.width, self.height), 3, dtype="f4")
        self.texA.use(location=0)

        # self.fbo = context.framebuffer(color_attachments=[self.texA])
        self.texB = self.context.texture((self.width, self.height), 3, dtype="f4")

        self.fboA = context.framebuffer(color_attachments=[self.texA])
        self.fboB = context.framebuffer(color_attachments=[self.texB])

        # self.frame_counter = 0
        self.cycle_counter = 0
        self.shader_rng_counter = 0


        program["STATIC_RENDER"].write(struct.pack("i", True))
        program["MAX_BOUNCES"].write(struct.pack("i", self.MAX_RAY_BOUNCES))
        program["RAYS_PER_PIXEL"].write(struct.pack("i", self.RAYS_PER_PIXEL))

        texture = context.texture((self.width, self.height), 3)
        texture.use(location=1)
        program["previousFrame"] = 1

        self.configure_buffers_and_uniforms(scene)

        # initialise the uniforms
        
    def configure_buffers_and_uniforms(self, scene:Scene):
        program = self.program
        spheres = scene.spheres
        context = self.context

        program["screenWidth"].write(struct.pack("i", self.width))
        program["spheresCount"].write(struct.pack("i", len(spheres)))

        sphere_buffer_binding = 1
        program["sphereBuffer"].binding = sphere_buffer_binding
        sphere_bytes = functions.iter_to_bytes(spheres)
        if len(sphere_bytes) > 0:
            sphere_buffer = context.buffer(sphere_bytes)
            sphere_buffer.bind_to_uniform_block(self.sphere_buffer_binding)

        triangles = scripts.numba_utils.classes.get_all_triangles_arr()

        program["meshCount"].write(struct.pack("i", len(scene.meshes)))

        print(f"Updating Triangle Positions...")
        def update_triangle_positions():
            for mesh in scene.meshes:
                if mesh.is_awaiting_mesh_update:
                    scripts.numba_utils.classes.update_triangles_to_csys(mesh.triangles, mesh.csys)
                    mesh.bvh_graph.flag_for_mesh_update()
                    mesh.unflag_for_mesh_update()
        timer(update_triangle_positions)()

        def update_graph_aabbs():
            for graph in BVHGraph.ALL:
                graph.update_graph_node_aabbs_for_changed_triangles(force=False)
                
        timer(update_graph_aabbs)()

        print(f"Loading triangles into SSBO...")
        triangle_data = timer(scripts.numba_utils.classes.triangles_to_array)(triangles)
        triangle_bytes = triangle_data.tobytes()
        self.triangles_ssbo = context.buffer(triangle_bytes)
        triangles_ssbo_binding = 9
        self.triangles_ssbo.bind_to_storage_buffer(binding=triangles_ssbo_binding)
        print(f"Loading triangles complete.")

        print(f"Loading meshes into SSBO...")
        # mesh_buffer = context.buffer(functions.iter_to_bytes(scene.meshes))
        mesh_buffer = context.buffer(functions.iter_to_bytes(Mesh.ALL))
        mesh_buffer_binding = 10
        program["meshBuffer"].binding = mesh_buffer_binding
        mesh_buffer.bind_to_uniform_block(mesh_buffer_binding)
        print(f"Loading meshes complete.")

        # sky_color = Vector3((131, 200, 228), dtype="f4") / 255
        # program["skyColor"].write(struct.pack("3f", *sky_color))

        material_buffer = context.buffer(self.app.display_scene.atmosphere_material.tobytes())
        material_buffer_binding = 11
        program["materialBuffer"].binding = material_buffer_binding
        material_buffer.bind_to_uniform_block(material_buffer_binding)

        print(f"Loading bvh graph data into SSBO...")
        mesh_bvh_graph_bytes = functions.iter_to_bytes(BVHParentNode.ALL)
        self.mesh_bvh_graph_ssbo = context.buffer(mesh_bvh_graph_bytes)
        self.mesh_bvh_graph_ssbo.bind_to_storage_buffer(binding=12)
        print(f"Loading bvh graph data complete.")

        print(f"Loading bvh_tri_id list into SSBO...")
        node_triangle_id_arr_bytes = np.array(BVHGraph.BVH_TRI_ID_LIST_GLOBAL, dtype=np.int32).tobytes()
        self.bvh_tri_ids_buffer = context.buffer(node_triangle_id_arr_bytes)
        self.bvh_tri_ids_buffer.bind_to_storage_buffer(binding=13)
        print(f"Loading bvh_tri_id list complete.")

        pass

        # program["chunksx"].write(struct.pack("i", self.app.CHUNKSX))
        # program["chunksy"].write(struct.pack("i", self.app.CHUNKSY))
        program["chunksx"].write(struct.pack("i", 1))
        program["chunksy"].write(struct.pack("i", 1))

        program["MAX_CYCLES"].write(struct.pack("i", self.app.MAX_CYCLES))

        self.app.is_animating = True
        if not "app_frame_counter" in self.__dir__():
            self.app_frame_counter = 0
        pass

    def calculate_frame_via_chunking(self, scene: Scene, chunksx = 1, chunksy = 1, force_flip_once=True):
        import gc

        gc.disable()
        vao = self.vao
        program = self.program
        context = self.context
        cam = scene.cam

        time_of_last_render = time.time_ns()
        time_between_renders = 1 / 30 * 1e9
        _t = time_of_last_render

        t1 = None
        t2 = None
        t3 = None

        has_flipped = False

        def render_to_screen(tex):
            nonlocal t1, t2, t3, time_of_last_render, has_flipped
            t1 = time.time_ns()
            context.screen.use()
            # self.texA.use(location=0)
            tex.use(location=0)
            self.screen_prog["uTexture"].value = 0
            self.screen_vao.render(mode=moderngl.TRIANGLE_STRIP)
            t2 = time.time_ns()
            pygame.display.flip()
            t3 = time.time_ns()
            print(f"render: {(t2-t1)/1e6:.2f} ms | flip: {(t3-t2)/1e6:.2f} ms")
            print(f"render time: {(t2-t1) / 1e6} ms")
            time_of_last_render = _t
            has_flipped = True
            context.finish()


        def handle_pg_events():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

        program["chunksx"].write(struct.pack("i", chunksx))
        program["chunksy"].write(struct.pack("i", chunksy))
        for i in range(chunksx):
            for j in range(chunksy):
                program["chunkx"].write(struct.pack("i", i))
                program["chunky"].write(struct.pack("i", j))

                # use texB as the previous frame sampler
                self.texB.use(location=1)
                # render to fboA
                self.fboA.use()
                vao.render(mode=moderngl.TRIANGLE_STRIP)

                _t = time.time_ns()
                print(f"{i:03d}, {j:03d}: {(_t-time_of_last_render) / 1e6} ms")

                # if (_t - time_of_last_render) > time_between_renders:
                if True:
                    render_to_screen(self.texA)
                    handle_pg_events()

                context.finish()

                self.app.clock.tick(1000)

                # swap the textures at the end of rendering
                self.fboA, self.fboB = self.fboB, self.fboA
                self.texA, self.texB = self.texB, self.texA


                pass

        # force one render per cycle
        # if force_flip_once and (not has_flipped):
        #     render_to_screen(self.texA)
        #     handle_pg_events()
        #     context.finish()


        gc.enable()

        time.sleep(0.001)
        return

    def save_frame(self, fbo, frame, cycle):
        if not "target_dir" in self.__dir__():
            self.target_dir = (
                Path()
                / "renders/RayTracerDynamic"
                / f"{datetime.datetime.now().strftime("%Y.%m.%d_%H%M%S")}"
            )
            self.target_dir.mkdir(parents=True)

        # buffer = self.fboB.read(components=3, dtype="f1")
        buffer = fbo.read(components=3, dtype="f4")
        hdr_data = np.frombuffer(buffer, dtype=np.float32)
        hdr_data = hdr_data.reshape((self.height, self.width, 3))
        
        rgb = np.clip(hdr_data, 0.0, 1.0)
        rgb_flipped = np.flip(rgb, axis=0)
        rgb_display = rgb_flipped
        display_8bit = (rgb_display[..., :3] * 255).astype(np.uint8)
        import imageio
        imageio.imwrite(self.target_dir / f"f{frame:05}_c{cycle:05}.png", display_8bit, format="png")
        pass

    def calculate_frame_no_chunk_and_show(self):

        self.program["chunksx"].write(struct.pack("i", 1))
        self.program["chunksy"].write(struct.pack("i", 1))
        self.program["chunkx"].write(struct.pack("i", 0))
        self.program["chunky"].write(struct.pack("i", 0))

        self.texB.use(location=1)
        self.fboA.use()
        self.vao.render(mode=moderngl.TRIANGLE_STRIP)

        self.context.screen.use()
        self.texA.use(location=0)
        self.screen_prog["uTexture"].value = 0
        self.screen_vao.render(mode=moderngl.TRIANGLE_STRIP)

        self.fboA, self.fboB = self.fboB, self.fboA
        self.texA, self.texB = self.texB, self.texA

        pygame.display.flip()

    def calculate_frame(self, scene: Scene):
        import time
        vao = self.vao
        program = self.program
        context = self.context
        cam = scene.cam

        spheres = scene.spheres

        # framerate = 10
        # if not "time_of_last_frame_render_ns" in self.__dir__():
        #     self.time_of_last_frame_render_ns = 0
        # if (time.time_ns() - self.time_of_last_frame_render_ns) > 1 / framerate * 1e9:
        if self.app_frame_counter > 60:
            self.app_frame_counter = 0
        # self.time_of_last_frame_render_ns = time.time_ns()
        self.app_frame_counter += 1
        # get the frame number from the scene object
        self.app_frame_counter = SCENE.get_frame_number(scene, self.app_frame_counter)

        # animate based on the frame
        self.app.animate_frame(self.app_frame_counter)


        program["depthOfFieldStrength"].write(struct.pack("f", cam.depth_of_field_strength))
        program["antialiasStrength"].write(struct.pack("f", cam.antialias_strength))

        program["ViewParams"].write(cam.view_params.astype("f4"))   
        program["CamLocalToWorldMatrix"].write(cam.local_to_world_matrix.astype("f4"))
        program["CamGlobalPos"].write(cam.csys.pos.astype("f4"))

        program["MAX_BOUNCES"].write(struct.pack("i", scene.cam.bounces_per_ray))
        program["RAYS_PER_PIXEL"].write(struct.pack("i", scene.cam.rays_per_pixel))

        def render_no_chunk():
            scene.cam.cycle_counter = 0
            program["frameNumber"].write(struct.pack("I", scene.cam.cycle_counter))
            self.calculate_frame_no_chunk_and_show()

        def render_with_chunk():
            scene.cam.cycle_counter = 0
            # scene.cam.chunksy = 2
            # scene.cam.chunksx = 2
            for _ in range(scene.cam.passes_per_frame):
                # if the watchdog updates between frames, allow it to break
                if self.app.is_requesting_exit.is_set():
                    return
                scene.cam.cycle_counter += 1
                program["frameNumber"].write(struct.pack("I", scene.cam.cycle_counter))
                self.calculate_frame_via_chunking(scene, chunksx=scene.cam.chunksx, chunksy=scene.cam.chunksy, force_flip_once=True)
                print(f"cycle={scene.cam.cycle_counter}")
                # if not (scene.cam.cycle_counter % 10):
                self.save_frame(self.fboB, self.app_frame_counter, cam.cycle_counter)
                # time.sleep(0.01)
                pass

        # render_no_chunk()
        render_with_chunk()

        if self.is_scene_static:
            scene.cam.cycle_counter += 1
            program["frameNumber"].write(struct.pack("I", self.cycle_counter))
        else:
            self.is_scene_static = True
            scene.cam.cycle_counter = 0
            program["frameNumber"].write(struct.pack("I", 0))
            pass

        print(self.cycle_counter)

        self.context.gc()

    def handle_event(
        self,
        event,
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
            # mouse_ray = functions.convert_screen_coords_to_camera_ray(
            #     mouse_x,
            #     mouse_y,
            #     app.screen.get_width(),
            #     app.screen.get_height(),
            #     cam=scene.cam,
            # )
            # hits = functions.rayScene(mouse_ray, scene)
            # hits = [h for h in hits if isinstance(h[1], Triangle | Sphere)]
            # if hits:
            #     # mouse_sphere.pos = hits[0][2]
            #     pass

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


def generic_camera_event_handler(
    event,
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


def tone_map(color):
    # color: H x W x 3 or 4
    return color / (color + 1.0)

def gamma_correct(color, gamma=2.2):
    return np.power(color, 1.0 / gamma)


pass
