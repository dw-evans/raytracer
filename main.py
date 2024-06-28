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
)

from scripts.functions import (
    iter_to_bytes,
    buffer_to_image_float16,
    buffer_to_image,
)


os.chdir(Path(__file__).parent)


WINDOW_HEIGHT = 1080
# ASPECT_RATIO = 16.0 / 9.0
SCALE_FACTOR = 1

STATIC_RENDER = True
STATIC_RENDER_ANIMATION = False

DYNAMIC_RENDER_FRAMERATE = 144

MAX_RAY_BOUNCES = 4
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

h = WINDOW_HEIGHT
w = int(h * cam.aspect)
pygame.init()

screen = pygame.display.set_mode(
    (w * SCALE_FACTOR, h * SCALE_FACTOR),
    pygame.OPENGL | pygame.DOUBLEBUF,
)

clock = pygame.time.Clock()

# get the vertex and fragment shaders
shader_vertex = ""
shader_fragment = ""

with open("shaders/raytracer.fs.glsl") as f:
    shader_fragment = f.read()

with open("shaders/raytracer.vs.glsl") as f:
    shader_vertex = f.read()

ctx = moderngl.create_context()


program = ctx.program(
    vertex_shader=shader_vertex,
    fragment_shader=shader_fragment,
)

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
        # pos=Vector3((0, 0, -30), dtype="f4"),
        pos=Vector3((2, 3, 4), dtype="f4"),
        radius=1,
        material=material_light_source_1,
    ),
    Sphere(
        # pos=Vector3((0, 0, -30), dtype="f4"),
        pos=Vector3((-2, 3, 4), dtype="f4"),
        radius=1,
        material=material_light_source_1,
    ),
    Sphere(
        pos=Vector3((1, 0.5, 0), dtype="f4"),
        radius=0.5,
        material=material_light_source_2,
    ),
]


triangles: list[Triangle] = [
    # Triangle(
    #     Vector3((-1, -0.5, 5), dtype="f4"),
    #     Vector3((-1, -0, 8), dtype="f4"),
    #     Vector3((1, -0.8, 5), dtype="f4"),
    #     material=material_plain_4,
    # ).translate(Vector3((-1.05, 0.0, 0.0))),
    # Triangle(
    #     Vector3((-1, -0, 8), dtype="f4"),
    #     Vector3((1, -0, 8), dtype="f4"),
    #     Vector3((1, -0.8, 5), dtype="f4"),
    #     material=material_plain_4,
    # ).translate(Vector3((-1.05, 0.0, 0.0))),
    # Triangle(
    #     Vector3((-1, -0.8, 5), dtype="f4"),
    #     Vector3((-1, -0, 8), dtype="f4"),
    #     Vector3((1, -0.8, 5), dtype="f4"),
    #     material=material_plain_5,
    # ).translate(Vector3((1.05, 0.0, 0.0))),
    # Triangle(
    #     Vector3((-1, -0, 8), dtype="f4"),
    #     Vector3((1, 0.5, 8), dtype="f4"),
    #     Vector3((1, -0.8, 5), dtype="f4"),
    #     material=material_plain_5,
    # ).translate(Vector3((1.05, 0.0, 0.0))),
]

# changing the winding direction turns the pixels black. maybe this should be transparent


# stl_file = Path() / "objects/warped_cube.stl"
stl_file = Path() / "objects/monkey.stl"

msh0 = Mesh.from_stl(
    stl_file,
    material=material_plain_4,
)
scene.meshes.append(msh0)

msh1 = Mesh.from_stl(
    stl_file,
    material=material_plain_5,
)
scene.meshes.append(msh1)

msh2 = Mesh.from_stl(
    stl_file,
    material=material_plain_5,
)
scene.meshes.append(msh2)

buffer1 = ctx.buffer(vertices.tobytes())

vao = ctx.vertex_array(
    program,
    [
        (buffer1, "3f", "position"),
    ],
)

# program["STATIC_RENDER"].value = STATIC_RENDER
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

cam.pos.z -= 4
triangles = scene.triangles
n_triangles = len(triangles)

program["triCount"].write(struct.pack("i", n_triangles))

# data = iter_to_bytes(triangles[:100])
# tri_buffer = ctx.buffer(data)
# tri_buffer_binding = 2
# tri_buffer.bind_to_uniform_block(tri_buffer_binding)
# program["triBuffer"].binding = tri_buffer_binding

program["meshCount"].write(struct.pack("i", len(scene.meshes)))

msh0.csys.translate(Vector3((-0.5, 0.0, 6.0)))
msh0.csys.Rz(180)

msh1.csys.translate(Vector3((1.0, 0.5, 8.0)))
msh1.csys.Rz(180)

msh2.csys.translate(Vector3((-2.0, 0.0, 8.0)))
msh2.csys.Rz(180)

tri_buffer_length_max = 455

triangles_ssbo = ctx.buffer(
    iter_to_bytes(
        [t.update_pos_with_mesh2() for t in scene.triangles],
    )
)
triangles_ssbo_binding = 9
triangles_ssbo.bind_to_storage_buffer(binding=triangles_ssbo_binding)
# program["triBuffer"].binding = triangles_ssbo_binding

# create the multiple buffers to store the triangle information
# for i in range(12):
#     start = min(i * tri_buffer_length_max, n_triangles)
#     stop = min((i + 1) * tri_buffer_length_max, n_triangles + 1)
#     if start - stop == -1:
#         break
#     # data = iter_to_bytes(
#     #     triangles[i * tri_buffer_length_max : (i + 1) * tri_buffer_length_max]
#     # )
#     # write the data using the transformed triangles of the mesh
#     data = iter_to_bytes(
#         [
#             t.update_pos_with_mesh2()
#             for t in triangles[
#                 i * tri_buffer_length_max : (i + 1) * tri_buffer_length_max
#             ]
#         ]
#     )

#     tri_buffer = ctx.buffer(data)
#     tri_buffer_binding = 3 + i
#     tri_buffer.bind_to_uniform_block(tri_buffer_binding)

#     program[f"triBuffer{i}"].binding = tri_buffer_binding
#     pass


mesh_buffer = ctx.buffer(iter_to_bytes(scene.meshes))
mesh_buffer_binding = 10
program["meshBuffer"].binding = mesh_buffer_binding

# tri_bytes = iter_to_bytes(triangles)
# tri_buffer = ctx.buffer(tri_bytes)
# tri_ssbo = ctx.buffer(tri_bytes)
# tri_ssbo.bind_to_storage_buffer(tri_buffer_binding)
# program["triBuffer"].binding = tri_buffer_binding


frame_counter = 0
cycle_counter = 0
shader_rng_counter = 0

running = True

sp0 = spheres[2]
x0, y0, z0 = sp0.pos

sp1 = spheres[-1]
x1, y1, z1 = sp1.pos

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
                        cam.pos.z += 1 / DYNAMIC_RENDER_FRAMERATE * SPEED
                    if key_state[pygame.K_s]:
                        cam.pos.z += -1 / DYNAMIC_RENDER_FRAMERATE * SPEED
                    if key_state[pygame.K_d]:
                        cam.pos.x += -1 / DYNAMIC_RENDER_FRAMERATE * SPEED
                    if key_state[pygame.K_a]:
                        cam.pos.x += 1 / DYNAMIC_RENDER_FRAMERATE * SPEED
                    if key_state[pygame.K_q]:
                        cam.pos.z += -1 / DYNAMIC_RENDER_FRAMERATE * SPEED
                    if key_state[pygame.K_e]:
                        cam.pos.z += -1 / DYNAMIC_RENDER_FRAMERATE * SPEED

                if MOUSE_ENABLED:
                    MOUSE_SENSITIVITY = 1.0
                    pygame.event.set_grab(True)
                    pygame.mouse.set_visible(False)
                    mouse_dx, mouse_dy = pygame.mouse.get_rel()
                    if mouse_dx:
                        cam.yaw += (
                            0.01
                            * MOUSE_SENSITIVITY
                            * mouse_dx
                            * 1
                            / DYNAMIC_RENDER_FRAMERATE
                        )
                    if mouse_dy:
                        cam.roll += (
                            0.01
                            * MOUSE_SENSITIVITY
                            * mouse_dy
                            * 1
                            / DYNAMIC_RENDER_FRAMERATE
                        )

            if STATIC_RENDER:
                time = 0
            else:
                time = pygame.time.get_ticks() / np.float32(1000.0)

            program["frameNumber"].write(struct.pack("I", shader_rng_counter))

            # sp0.pos.x = x0 + 0.5 * sin(time / 3)
            # sp0.pos.y = y0 + 0.5 * sin(time / 5)
            # sp0.pos.z = z0 + 0.5 * sin(time / 7)

            # sp1.pos.x = 0 + 2 * sin(time / 3)
            # sp1.pos.y = 2.0
            # sp1.pos.z = 6.0 + 2 * sin(time / 3)

            program["ViewParams"].write(cam.view_params.astype("f4"))
            program["CamLocalToWorldMatrix"].write(
                cam.local_to_world_matrix.astype("f4")
            )
            program["CamGlobalPos"].write(cam.pos.astype("f4"))

            sphere_bytes = iter_to_bytes(spheres)
            sphere_buffer = ctx.buffer(sphere_bytes)
            sphere_buffer.bind_to_uniform_block(sphere_buffer_binding)

            # tri_bytes = iter_to_bytes(triangles)
            # tri_buffer = ctx.buffer(tri_bytes)
            # tri_buffer.bind_to_uniform_block(tri_buffer_binding)

            vao.render(mode=moderngl.TRIANGLE_STRIP)

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


elif STATIC_RENDER and STATIC_RENDER_ANIMATION:
    try:
        """
        ffmpeg build details here
        "C:\Program Files\GNU Octave\Octave-6.4.0\mingw64\bin\ffmpeg.exe" -r 24 -i %06d.png -vf "fps=24,format=yuv420p" output.mp4
        """
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            time = frame_counter * (dt + 1)

            program["frameNumber"].write(struct.pack("I", shader_rng_counter))

            program["ViewParams"].write(cam.view_params.astype("f4"))
            program["CamLocalToWorldMatrix"].write(
                cam.local_to_world_matrix.astype("f4")
            )
            program["CamGlobalPos"].write(cam.pos.astype("f4"))

            sp1.pos.x = 0 + 3 * sin(time / 3)
            sp1.pos.y = 2.0
            sp1.pos.z = 6.0 + 3 * sin(time / 3)

            sphere_bytes = iter_to_bytes(spheres)
            sphere_buffer = ctx.buffer(sphere_bytes)
            sphere_buffer.bind_to_uniform_block(sphere_buffer_binding)

            tri_bytes = iter_to_bytes(triangles)
            tri_buffer = ctx.buffer(tri_bytes)
            tri_buffer.bind_to_uniform_block(tri_buffer_binding)

            vao.render(mode=moderngl.TRIANGLE_STRIP)

            render_data = ctx.screen.read(components=3, dtype="f1")
            texture.write(render_data)

            shader_rng_counter += 1
            cycle_counter += 1

            pygame.display.flip()

            if cycle_counter > STATIC_RENDER_CYCLES_PER_FRAME:
                # extract the image
                render_data2 = ctx.screen.read(components=3, dtype="f2")
                # img = buffer_to_image_float16(render_data2, (w, h))
                img = buffer_to_image(render_data, (w, h))
                # save the image to the renders folder
                img.save(dir_render / f"{frame_counter:06}.png")
                # reset the seed
                shader_rng_counter = 0
                # reset the cycle counter
                cycle_counter = 0
                # reset the texture for the next frame... maybe

                frame_counter += 1

            if frame_counter > n_frames:
                print("Render complete")
                running = False

        pygame.quit()

    except KeyboardInterrupt:
        pass
    finally:
        sys.exit()
