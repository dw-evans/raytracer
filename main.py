from __future__ import annotations

import pygame
import moderngl
import os
import sys

import numpy as np
from numpy import sin, cos, tan, radians
from array import array
from pathlib import Path
import struct
from stl import mesh
import pyrr
from pyrr import Vector3, Matrix44, Vector4

from typing import Protocol
from typing import Iterable

from PIL import Image
import datetime
from pathlib import Path


from scripts.classes import (
    ByteableObject,
    Triangle,
    Camera,
    Mesh,
    Material,
    Sphere,
    Scene,
    Csys,
    Ray,
)

import threading
import queue

from scripts.functions import (
    iter_to_bytes,
    buffer_to_image_float16,
    buffer_to_image,
    convert_screen_coords_to_camera_ray,
    rayScene,
)


class ShaderProgram:
    DUMB = "dumb"
    RAYTRACER = "raytracer"


os.chdir(Path(__file__).parent)


PROGRAM = ShaderProgram.RAYTRACER

WINDOW_HEIGHT = 270
ASPECT_RATIO = 16.0 / 9.0
SCALE_FACTOR = 1

STATIC_RENDER = False
STATIC_RENDER_ANIMATION = False

DYNAMIC_RENDER_FRAMERATE = 1

MAX_RAY_BOUNCES = 2
RAYS_PER_PIXEL = 1

STATIC_RENDER_FRAMERATE = 144
STATIC_RENDER_CYCLES_PER_FRAME = 256
STATIC_RENDER_TIME_DURATION = 2.0

dt = 1 / STATIC_RENDER_FRAMERATE
n_frames = STATIC_RENDER_FRAMERATE * STATIC_RENDER_TIME_DURATION


wd = Path(__file__).parent

date = datetime.datetime.now()

if STATIC_RENDER and STATIC_RENDER_ANIMATION:
    dir_render = wd / "renders" / date.strftime("%Y.%m.%d_%H%M%S")
    dir_render.mkdir()

MOUSE_ENABLED = False
KEYBOARD_ENABLED = False


scene = Scene()
cam = scene.cam
cam.aspect = ASPECT_RATIO

h = WINDOW_HEIGHT
w = int(h * cam.aspect)


vertices = np.array(
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

material_plain_1 = Material(
    Vector4((1.0, 0.5, 0.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    0.0,
    smoothness=0.0,
)
material_plain_2 = Material(
    Vector4((1.0, 1.0, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    0.0,
    smoothness=0.6,
)
material_plain_3 = Material(
    Vector4((0.5, 0.0, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    0.0,
    smoothness=0.0,
)
material_plain_4 = Material(
    Vector4((1.0, 0, 0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    0.3,
    smoothness=0.2,
)
material_plain_5 = Material(
    Vector4((0.0, 1.0, 0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    0.3,
    smoothness=0.2,
)
material_light_source_1 = Material(
    Vector4((0.0, 0.0, 0.0, 1.0), dtype="f4"),
    Vector3((1, 1, 1), dtype="f4"),
    5.0,
)
material_light_source_2 = Material(
    Vector4((0.0, 0.0, 0.0, 1.0), dtype="f4"),
    Vector3((1, 1, 1), dtype="f4"),
    5.0,
)
highlight_material = Material(
    Vector4((0.3, 0.6, 0.8, 1.0), dtype="f4"),
    Vector3((0, 0, 0), dtype="f4"),
    0.0,
    smoothness=0.5,
)

spheres = [
    Sphere(
        pos=Vector3((0.0, -1001.0, 8), dtype="f4"),
        radius=1000.0,
        material=material_plain_3,
    ),
    Sphere(
        pos=Vector3((0.0, 3.0, 22), dtype="f4"),
        radius=10.0,
        material=material_plain_1,
    ),
    Sphere(
        pos=Vector3((2, 3, 4), dtype="f4"),
        radius=1,
        material=material_light_source_1,
    ),
    Sphere(
        pos=Vector3((-2, 3, 4), dtype="f4"),
        radius=1,
        material=material_light_source_1,
    ),
    Sphere(
        pos=Vector3((1, 0.5, 0), dtype="f4"),
        radius=0.5,
        material=material_light_source_2,
    ),
    Sphere(
        pos=Vector3((0, 0, 0), dtype="f4"),
        radius=0.2,
        material=material_plain_1,
    ),
]
scene.spheres = spheres

stl_file = Path() / "objects/monkey.stl"

msh0 = Mesh.from_stl(
    stl_file,
    material=material_plain_4,
)
scene.meshes.append(msh0)
msh0.csys.tp(Vector3((0, 0.0, 6.0)))
msh0.csys.Rzg(180)

msh1 = Mesh.from_stl(
    stl_file,
    material=material_plain_4,
)
scene.meshes.append(msh1)

msh1.csys.tp(Vector3((-1.5, 1.0, 9.0)))
msh1.csys.Rzg(100)

msh2 = Mesh.from_stl(
    stl_file,
    material=material_plain_4,
)
scene.meshes.append(msh2)
msh2.csys.tp(Vector3((2, 0.0, 8.0)))
msh2.csys.Rzg(90)
SELECTED_MESH_ID = -1


modification_waiting = False


def main_loop():
    global SELECTED_MESH_ID
    global triangles_ssbo
    global ctx
    global modification_waiting
    global modify_command

    pygame.init()

    screen = pygame.display.set_mode(
        (w * SCALE_FACTOR, h * SCALE_FACTOR),
        pygame.OPENGL | pygame.DOUBLEBUF,
    )
    clock = pygame.time.Clock()

    with open(f"shaders/{PROGRAM}.fs.glsl") as f:
        shader_fragment = f.read()

    with open(f"shaders/{PROGRAM}.vs.glsl") as f:
        shader_vertex = f.read()

    ctx = moderngl.create_context()

    program = ctx.program(
        vertex_shader=shader_vertex,
        fragment_shader=shader_fragment,
    )

    buffer1 = ctx.buffer(vertices.tobytes())

    vao = ctx.vertex_array(
        program,
        [
            (buffer1, "3f", "position"),
        ],
    )

    if PROGRAM == ShaderProgram.RAYTRACER:
        program["STATIC_RENDER"].write(struct.pack("i", STATIC_RENDER))
        program["MAX_BOUNCES"].write(struct.pack("i", MAX_RAY_BOUNCES))
        program["RAYS_PER_PIXEL"].write(struct.pack("i", RAYS_PER_PIXEL))

        texture = ctx.texture((w, h), 3)
        texture.use(location=1)
        program["previousFrame"] = 1
        render_data = b"\x00" * w * h * 3

        # initialise the uniforms
        program["screenWidth"].write(struct.pack("i", w))
        program["screenHeight"].write(struct.pack("i", h))

        program["spheresCount"].write(struct.pack("i", len(spheres)))
        sphere_buffer_binding = 1
        program["sphereBuffer"].binding = sphere_buffer_binding

        # cam.pos.z += -4
        # cam.csys.pos.z += -4

        triangles = scene.triangles
        n_triangles = len(triangles)
        program["triCount"].write(struct.pack("i", n_triangles))

        program["meshCount"].write(struct.pack("i", len(scene.meshes)))

        # tri_buffer_length_max = 455

        triangles_ssbo = ctx.buffer(
            iter_to_bytes(
                [t.update_pos_with_mesh2() for t in scene.triangles],
            )
        )
        triangles_ssbo_binding = 9
        triangles_ssbo.bind_to_storage_buffer(binding=triangles_ssbo_binding)
        # program["triBuffer"].binding = triangles_ssbo_binding

        mesh_buffer = ctx.buffer(iter_to_bytes(scene.meshes))
        mesh_buffer_binding = 10
        program["meshBuffer"].binding = mesh_buffer_binding
        mesh_buffer.bind_to_uniform_block(mesh_buffer_binding)

        sky_color = Vector3((131, 200, 228), dtype="f4") / 255
        ground_color = Vector3((74, 112, 45), dtype="f4") / 255

        program["skyColor"].write(struct.pack("3f", *sky_color))

        material_buffer = ctx.buffer(iter_to_bytes([highlight_material]))
        material_buffer_binding = 11
        program["materialBuffer"].binding = material_buffer_binding
        material_buffer.bind_to_uniform_block(material_buffer_binding)

    elif PROGRAM == ShaderProgram.DUMB:
        # program["STATIC_RENDER"].write(struct.pack("i", STATIC_RENDER))
        # program["MAX_BOUNCES"].write(struct.pack("i", MAX_RAY_BOUNCES))
        # program["RAYS_PER_PIXEL"].write(struct.pack("i", RAYS_PER_PIXEL))

        # texture = ctx.texture((w, h), 3)
        # texture.use(location=1)
        # program["previousFrame"] = 1
        # render_data = b"\x00" * w * h * 3

        # initialise the uniforms
        # program["screenWidth"].write(struct.pack("i", w))
        # program["screenHeight"].write(struct.pack("i", h))

        program["spheresCount"].write(struct.pack("i", len(spheres)))
        sphere_buffer_binding = 1
        program["sphereBuffer"].binding = sphere_buffer_binding

        # cam.pos.z += -4
        # cam.csys.pos.z += -4

        triangles = scene.triangles
        n_triangles = len(triangles)
        program["triCount"].write(struct.pack("i", n_triangles))

        program["meshCount"].write(struct.pack("i", len(scene.meshes)))

        # tri_buffer_length_max = 455

        triangles_ssbo = ctx.buffer(
            iter_to_bytes(
                [t.update_pos_with_mesh2() for t in scene.triangles],
            )
        )
        triangles_ssbo_binding = 9
        triangles_ssbo.bind_to_storage_buffer(binding=triangles_ssbo_binding)
        # program["triBuffer"].binding = triangles_ssbo_binding

        mesh_buffer = ctx.buffer(iter_to_bytes(scene.meshes))
        mesh_buffer_binding = 10
        program["meshBuffer"].binding = mesh_buffer_binding
        mesh_buffer.bind_to_uniform_block(mesh_buffer_binding)

        sky_color = Vector3((131, 200, 228), dtype="f4") / 255
        ground_color = Vector3((74, 112, 45), dtype="f4") / 255

        program["skyColor"].write(struct.pack("3f", *sky_color))

    frame_counter = 0
    cycle_counter = 0
    shader_rng_counter = 0

    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)

    mouse_sphere = spheres[-1]

    running = True
    if not STATIC_RENDER_ANIMATION:
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        pygame.quit()
                        sys.exit()

                    if KEYBOARD_ENABLED:
                        SPEED = 2.0
                        key_state = pygame.key.get_pressed()
                        if key_state[pygame.K_w]:
                            cam.csys.pos.z += 1 / DYNAMIC_RENDER_FRAMERATE * SPEED
                        if key_state[pygame.K_s]:
                            cam.csys.pos.z += -1 / DYNAMIC_RENDER_FRAMERATE * SPEED
                        if key_state[pygame.K_d]:
                            cam.csys.pos.x += -1 / DYNAMIC_RENDER_FRAMERATE * SPEED
                        if key_state[pygame.K_a]:
                            cam.csys.pos.x += 1 / DYNAMIC_RENDER_FRAMERATE * SPEED
                        if key_state[pygame.K_q]:
                            cam.csys.pos.z += -1 / DYNAMIC_RENDER_FRAMERATE * SPEED
                        if key_state[pygame.K_e]:
                            cam.csys.pos.z += -1 / DYNAMIC_RENDER_FRAMERATE * SPEED

                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    # print(mouse_x, mouse_y)

                    mouse_ray = convert_screen_coords_to_camera_ray(
                        mouse_x,
                        mouse_y,
                        screen.get_width(),
                        screen.get_height(),
                        cam=scene.cam,
                    )

                    hits = rayScene(mouse_ray, scene)
                    hits = [h for h in hits if isinstance(h[1], Triangle)]
                    if hits:
                        # print(
                        #     f"n_hits={len(hits)}, {', '.join((str(type(x[1])) for x in hits))}"
                        # )
                        mouse_sphere.pos = hits[0][2]

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if hits:
                            obj = hits[0][1]
                            if isinstance(obj, Triangle):
                                SELECTED_MESH_ID = obj.parent.mesh_index
                                program["selectedMeshId"].write(
                                    struct.pack("i", SELECTED_MESH_ID)
                                )
                                # print(f"selectedMeshId = {SELECTED_MESH_ID}")
                        else:
                            SELECTED_MESH_ID = -1
                            program["selectedMeshId"].write(
                                struct.pack("i", SELECTED_MESH_ID)
                            )

                            # print(f"selectedMeshId = {SELECTED_MESH_ID}")

                if STATIC_RENDER:
                    time = 0
                else:
                    time = pygame.time.get_ticks() / np.float32(1000.0)

                if PROGRAM == ShaderProgram.RAYTRACER:
                    program["frameNumber"].write(struct.pack("I", shader_rng_counter))

                program["ViewParams"].write(cam.view_params.astype("f4"))
                program["CamLocalToWorldMatrix"].write(
                    cam.local_to_world_matrix.astype("f4")
                )
                program["CamGlobalPos"].write(cam.csys.pos.astype("f4"))

                sphere_bytes = iter_to_bytes(spheres)
                sphere_buffer = ctx.buffer(sphere_bytes)
                sphere_buffer.bind_to_uniform_block(sphere_buffer_binding)

                # tri_bytes = iter_to_bytes(triangles)
                # tri_buffer = ctx.buffer(tri_bytes)
                # tri_buffer.bind_to_uniform_block(tri_buffer_binding)

                if modification_waiting:
                    modify_command()
                    modification_waiting = False
                

                pass
                # triangles_ssbo_binding = 9
                # triangles_ssbo.bind_to_storage_buffer(binding=triangles_ssbo_binding)

                vao.render(mode=moderngl.TRIANGLE_STRIP)

                if PROGRAM == ShaderProgram.RAYTRACER:
                    render_data = ctx.screen.read(components=3, dtype="f1")
                    texture.write(render_data)

                shader_rng_counter += 1
                cycle_counter += 1

                pygame.display.flip()
                clock.tick(DYNAMIC_RENDER_FRAMERATE)

                pass

            # extract the image
            render_data2 = ctx.screen.read(components=3, dtype="f2")
            img = buffer_to_image_float16(render_data2, (w, h))
            # save the image to the renders folder
            img.save(dir_render / f"{frame_counter:05}.png")

            shader_rng_counter = 0
            cycle_counter = 0

        except KeyboardInterrupt:
            pass



def listener_loop():
    global SELECTED_MESH_ID
    global triangles_ssbo
    global ctx
    global modify_command
    global modification_waiting

    import regex as re

    valid_methods = dir(Csys)

    while True:

        command = input("Enter your command")

        print(f"selected index = {SELECTED_MESH_ID}")
        print([m.mesh_index for m in scene.meshes])

        try:
            msh: Mesh = [m for m in scene.meshes if m.mesh_index == SELECTED_MESH_ID][0]
        except IndexError:
            print("Mesh not valid")
            continue

        match = re.match(r"(\w*\D)(?:\s*)((?:\d+)(?:.\d*))", command)

        if not match:
            continue

        if not match.group(1).capitalize() in valid_methods:
            continue

        print(f"modifying mesh with command `{command}`")

        msh.csys.__getattribute__(match.group(1).capitalize())(float(match.group(2)))



        modify_command = lambda: triangles_ssbo.write(
            iter_to_bytes(
                [t.update_pos_with_mesh2() for t in scene.triangles],
            )
        )
        modification_waiting = True

        pass


def main():
    t1 = threading.Thread(target=main_loop)
    t2 = threading.Thread(target=listener_loop)

    t1.start()
    t2.start()


if __name__ == "__main__":
    main()
