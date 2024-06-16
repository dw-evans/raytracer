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

import pyrr

from pyrr import Vector3, Matrix44, Vector4

from typing import Protocol


# from PIL import Image


os.chdir(Path(__file__).parent)


class HitInfo:
    def __init__(self) -> None:
        self.didHit: bool = False
        self.dst: float = np.inf
        self.hitPoint = np.array([0.0, 0.0, 0.0])
        self.normal = np.array([0.0, 0.0, 0.0])


class Ray:
    def __init__(self) -> None:
        self.origin = np.array([0.0, 0.0, 0.0])
        self.dir = np.array([0.0, 0.0, 0.0])


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def raySphere(ray: Ray, spherePos: np.ndarray, rad: float) -> HitInfo:

    ret = HitInfo()

    offsetRayOrigin = ray.origin - spherePos

    a = np.dot(ray.dir, ray.dir)
    b = 2 * np.dot(offsetRayOrigin, ray.dir)
    c = np.dot(offsetRayOrigin, offsetRayOrigin) - rad * rad

    discriminant = b * b - 4 * a * c

    if discriminant >= 0:
        dst = (-b - np.sqrt(discriminant)) / (2 * a)

        if dst >= 0:
            ret.didHit = True
            ret.dst = dst
            ret.hitPoint = ray.origin + ray.dir * dst
            ret.normal = normalize(ret.hitPoint - spherePos)

    return ret


from typing import Iterable


def iter_to_bytes(iterable: Iterable[ByteableObject]) -> bytearray:
    ret = bytearray()
    for x in iterable:
        ret.extend(x.tobytes())
    return ret


import numpy as np
from PIL import Image


def buffer_to_image_float16(
    buffer: bytes, size: tuple[int, int], mode="RGB"
) -> Image.Image:
    buffer_np = np.frombuffer(buffer, np.float16).reshape((size[1], size[0], len(mode)))
    buffer_np = np.flipud(buffer_np)
    img = Image.fromarray(buffer_np, mode)
    return img


def buffer_to_image(buffer: bytes, size: tuple[int, int], mode="RGB") -> Image.Image:
    img = Image.frombuffer(mode, size, buffer)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


class ByteableObject(Protocol):
    def tobytes(self) -> bytes | bytearray: ...


class Sphere(ByteableObject):
    def __init__(self, pos: Vector3, radius: float, material: Material) -> None:
        self.pos = pos
        self.radius = radius
        self.material = material

    def tobytes(self):
        return struct.pack("3f f", *self.pos, self.radius) + self.material.tobytes()


class Triangle(ByteableObject):
    def __init__(
        self,
        posA: Vector3,
        posB: Vector3,
        posC: Vector3,
        material: Material,
        # normalA: Vector3,
        # normalB: Vector3,
        # normalC: Vector3,
    ) -> None:
        self.posA = posA
        self.posB = posB
        self.posC = posC
        self.material = material
        # lets just make the normals the basic way
        # self.normalA: Vector3 = self.normal
        # self.normalB: Vector3 = self.normal
        # self.normalC: Vector3 = self.normal

    @property
    def normal(self) -> Vector3:
        edge_ab = self.posB - self.posA
        edge_ac = self.posC - self.posA
        ret = edge_ab.cross(edge_ac)
        ret.normalize()
        return ret

    @property
    def normalA(self):
        return self.normal

    @property
    def normalB(self):
        return self.normal

    @property
    def normalC(self):
        return self.normal

    def tobytes(self) -> bytes | bytearray:
        # return struct.pack("3f 3f 3f", *self.posA, *self.posB, *self.posC)
        return (
            struct.pack(
                # "3f 3f 3f",
                "3f4x 3f4x 3f4x 3f4x 3f4x 3f4x",
                *self.posA.astype("f4"),
                *self.posB.astype("f4"),
                *self.posC.astype("f4"),
                *self.normal.astype("f4"),
                *self.normal.astype("f4"),
                *self.normal.astype("f4"),
            )
            + self.material.tobytes()
        )


class Material(ByteableObject):

    def __init__(
        self,
        color: Vector4,
        emissionColor: Vector3,
        emissionStrength: float,
        smoothness: float = 0.5,
    ) -> None:
        self.color = color
        self.emissionColor = emissionColor
        self.emissionStrength = emissionStrength
        self.smoothness = smoothness

    def tobytes(self):
        return struct.pack(
            "4f 3f f f12x",
            *self.color,
            *self.emissionColor,
            self.emissionStrength,
            self.smoothness,
        )


class Camera:
    def __init__(self) -> None:
        self.fov = 30.0  # degrees
        self.aspect = 16.0 / 9.0
        self.near_plane = 240.0
        self.pos = Vector3((0.0, 0.0, 0.0), dtype="f4")
        # self.orientation = Matrix44.identity()
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    @property
    def orientation(self):
        return Matrix44.from_eulers(
            [radians(x) for x in [self.roll, self.pitch, self.yaw]]
        )

    @property
    def plane_height(self):
        return self.near_plane * tan(radians(self.fov) * 0.5) * 2.0

    @property
    def plane_width(self):
        return self.plane_height * self.aspect

    @property
    def local_to_world_matrix(self):
        return self.orientation.inverse * Matrix44.from_translation(-1 * self.pos)

    @property
    def view_params(self):
        return Vector3(
            [self.plane_width, self.plane_height, self.near_plane],
        )

    def rotate(self, axis: str, rot_deg: int | float | Iterable[int | float]):
        if not isinstance(rot_deg, Iterable):
            degs = [rot_deg]

        if not len(axis) == len(degs):
            raise Exception("Incorrect length match")
        for axis, deg in zip(axis, degs):
            if axis == "x":
                self.orientation.from_x_rotation(radians(deg))
                pass
            elif axis == "y":
                self.orientation.from_y_rotation(radians(deg))
                pass
            elif axis == "z":
                self.orientation.from_z_rotation(radians(deg))
                pass


WINDOW_HEIGHT = 1080
# ASPECT_RATIO = 16.0 / 9.0

STATIC_RENDER = True
STATIC_RENDER_ANIMATION = True

DYNAMIC_RENDER_FRAMERATE = 4

MAX_RAY_BOUNCES = 4
RAYS_PER_PIXEL = 128

STATIC_RENDER_FRAMERATE = 24
STATIC_RENDER_CYCLES_PER_FRAME = 24
STATIC_RENDER_TIME_DURATION = 1.0

dt = 1 / STATIC_RENDER_FRAMERATE
n_frames = STATIC_RENDER_FRAMERATE * STATIC_RENDER_TIME_DURATION

import datetime
from pathlib import Path

wd = Path(__file__).parent

date = datetime.datetime.now()

dir_render = wd / "renders" / date.strftime("%Y.%m.%d_%H%M%S")
dir_render.mkdir()

MOUSE_ENABLED = False
KEYBOARD_ENABLED = False

cam = Camera()

h = WINDOW_HEIGHT
w = int(h * cam.aspect)
SCALE_FACTOR = 1
pygame.init()

screen = pygame.display.set_mode(
    (w * SCALE_FACTOR, h * SCALE_FACTOR),
    pygame.OPENGL | pygame.DOUBLEBUF,
)

clock = pygame.time.Clock()

# get the vertex and fragment shaders
shader_vertex = ""
shader_fragment = ""

with open("shaders/fs.glsl") as f:
    shader_fragment = f.read()

with open("shaders/vs.glsl") as f:
    shader_vertex = f.read()

ctx = moderngl.create_context()

ctx.enable(moderngl.BLEND)
ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

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


material1 = Material(
    Vector4((1.0, 0.5, 0.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    0.0,
    smoothness=0.0,
)
material2 = Material(
    Vector4((1.0, 1.0, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    0.0,
    smoothness=0.6,
)
material3 = Material(
    Vector4((0.5, 0.0, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    0.0,
    smoothness=0.0,
)
# light source
material4 = Material(
    Vector4((0.0, 0.0, 0.0, 1.0), dtype="f4"),
    Vector3((1, 1, 1), dtype="f4"),
    4.0,
)
# light source
material5 = Material(
    Vector4((0.0, 0.0, 0.0, 1.0), dtype="f4"),
    # Vector3((247 / 255, 214 / 255, 128 / 255), dtype="f4"),
    Vector3((1, 1, 1), dtype="f4"),
    1.0,
)


spheres = [
    Sphere(
        pos=Vector3((0.0, -1001.0, 8), dtype="f4"),
        radius=1000.0,
        material=material3,
    ),
    Sphere(
        pos=Vector3((0.0, 3.0, 22), dtype="f4"),
        radius=10.0,
        material=material1,
    ),
    Sphere(
        pos=Vector3((0.0, 3, 6), dtype="f4"),
        radius=1.0,
        material=material2,
    ),
    # Sphere(
    #     pos=Vector3((-3.0, 0.0, 8.0), dtype="f4"),
    #     radius=0.5,
    #     material=material2,
    # ),
    # Sphere(
    #     pos=Vector3((-2.0, 1.5, 5.0), dtype="f4"),
    #     radius=0.6,
    #     material=material3,
    # ),
    Sphere(
        pos=Vector3((0, 0, -30), dtype="f4"),
        radius=20,
        material=material4,
    ),
]


trimaterial = Material(
    Vector4((1.0, 0, 0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    0.0,
    smoothness=1,
)

# changing the winding direction turns the pixels black. maybe this should be transparent

triangles = [
    Triangle(
        Vector3((3, 2, 10.0), dtype="f4"),
        Vector3((1, 0, 10.0), dtype="f4"),
        Vector3((1, 2, 10.0), dtype="f4"),
        material=trimaterial,
    ),
    Triangle(
        Vector3((-1, 0, 10.0), dtype="f4"),
        Vector3((-3, 0, 10), dtype="f4"),
        Vector3((-3, 2, 10), dtype="f4"),
        material=trimaterial,
    ),
    Triangle(
        Vector3((1, -0.8, 5), dtype="f4"),
        Vector3((-1, -0, 8), dtype="f4"),
        Vector3((-1, -0.8, 5), dtype="f4"),
        material=trimaterial,
    ),
    Triangle(
        Vector3((1, -0, 8), dtype="f4"),
        Vector3((-1, -0, 8), dtype="f4"),
        Vector3((1, -0.8, 5), dtype="f4"),
        material=trimaterial,
    ),
]


sp1a, sp2a, sp3a = [
    Sphere(
        pos=triangles[0].posA,
        radius=0.1,
        material=material1,
    ),
    Sphere(
        pos=triangles[0].posB,
        radius=0.1,
        material=material1,
    ),
    Sphere(
        pos=triangles[0].posC,
        radius=0.1,
        material=material1,
    ),
]

sp1b, sp2b, sp3b = [
    Sphere(
        pos=triangles[1].posA,
        radius=0.1,
        material=material2,
    ),
    Sphere(
        pos=triangles[1].posB,
        radius=0.1,
        material=material2,
    ),
    Sphere(
        pos=triangles[1].posC,
        radius=0.1,
        material=material2,
    ),
]

spheres += [sp1a, sp2a, sp3a]
# spheres += [sp1b, sp2b, sp3b]

# sps = [
#     Sphere(
#         pos=triangles[2].posA,
#         radius=0.1,
#         material=material1,
#     ),
#     Sphere(
#         pos=triangles[2].posB,
#         radius=0.1,
#         material=material1,
#     ),
#     Sphere(
#         pos=triangles[2].posC,
#         radius=0.1,
#         material=material1,
#     ),
#     Sphere(
#         pos=triangles[3].posB,
#         radius=0.1,
#         material=material1,
#     ),
# ]


# spheres += sps

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

program["triCount"].write(struct.pack("i", len(triangles)))
tri_buffer_binding = 2
program["triBuffer"].binding = tri_buffer_binding


frame_counter = 0
cycle_counter = 0
shader_rng_counter = 0

running = True

sp = spheres[2]
x0, y0, z0 = sp.pos

if False:
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
                if STATIC_RENDER_ANIMATION:
                    time = shader_rng_counter * dt + dt
                    if frame_counter > n_frames:
                        break
                else:
                    time = 0
            else:
                time = pygame.time.get_ticks() / np.float32(1000.0)

            program["frameNumber"].write(struct.pack("I", shader_rng_counter))

            # spheres[4].pos.x = 2 * sin(time * 2.1)
            # spheres[4].pos.y = 5 + 3 * cos(time * 1.9)
            # spheres[4].pos.z = 8 + 5 * cos(time * 1.1)

            # spheres[1].pos.x = 1.5 * sin(time * 1)
            # spheres[1].pos.y = 0.5 * cos(time * 1)
            # spheres[1].pos.z = 16 + 0.5 * cos(time * 1)

            sp.pos.y = y0 + 1.0 * sin(time / 5)

            # cam.yaw = 15 * sin(time / 4)
            # cam.pitch = 20 * sin(time / 15)
            # cam.roll = 20 * sin(time / 8)

            # triangles[0].posA.x = 2.0 + 1 * sin(time / 3)
            # triangles[0].posB.x = -2.0 + 1 * sin(time / 7)
            # triangles[0].posC.x = -0.0 + 1 * sin(time / 5)

            # triangles[0].posA = Vector3((-2, 0, 11), dtype="f4")
            # triangles[0].posB = Vector3((2, 0, 9), dtype="f4")
            # triangles[0].posC = Vector3((0, 2 + 1.0 * sin(time), 10), dtype="f4")

            # cam.pos.z = 0.0 + 3.0 * sin(time / 8)
            cam.pos.z = -2

            program["ViewParams"].write(cam.view_params.astype("f4"))
            program["CamLocalToWorldMatrix"].write(
                cam.local_to_world_matrix.astype("f4")
            )
            program["CamGlobalPos"].write(cam.pos.astype("f4"))

            sphere_bytes = iter_to_bytes(spheres)
            # sphere_bytes += b"\x00" * (16 - len(sphere_bytes) % 16)
            sphere_buffer = ctx.buffer(sphere_bytes)
            sphere_buffer.bind_to_uniform_block(sphere_buffer_binding)

            tri_bytes = iter_to_bytes(triangles)
            # tri_bytes += b"\x00" * (16 - len(tri_bytes) % 16)
            tri_buffer = ctx.buffer(tri_bytes)
            tri_buffer.bind_to_uniform_block(tri_buffer_binding)

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

            time = frame_counter * (dt + 1)

            program["frameNumber"].write(struct.pack("I", shader_rng_counter))

            program["ViewParams"].write(cam.view_params.astype("f4"))
            program["CamLocalToWorldMatrix"].write(
                cam.local_to_world_matrix.astype("f4")
            )
            program["CamGlobalPos"].write(cam.pos.astype("f4"))

            sp.pos.x = x0 + 0.5 * sin(time / 2)
            sp.pos.y = y0 + 1.0 * sin(time / 3)
            sp.pos.z = z0 + 0.5 * sin(time / 5)

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
