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

from pyrr import Vector3, Matrix44, Vector4

from typing import Protocol


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


class ByteableObject(Protocol):
    def tobytes(self) -> bytes | bytearray: ...


class Sphere(ByteableObject):
    def __init__(self, pos: Vector3, radius: float, material: Material) -> None:
        self.pos = pos
        self.radius = radius
        self.material = material

    def tobytes(self):
        return struct.pack("3f f", *self.pos, self.radius) + self.material.tobytes()


class Material(ByteableObject):

    def __init__(
        self,
        color: Vector4,
        emissionColor: Vector3,
        emissionStrength: float,
    ) -> None:
        self.color = color
        self.emissionColor = emissionColor
        self.emissionStrength = emissionStrength

    def tobytes(self):
        return struct.pack(
            "4f 3f f", *self.color, *self.emissionColor, self.emissionStrength
        )
        # return struct.pack(
        #     "4f",
        #     *self.color,
        # )


class Camera:
    def __init__(self) -> None:
        self.fov = 30.0  # degrees
        self.aspect = 16.0 / 9.0
        self.near_plane = 240.0
        self.pos = Vector3((0.0, 0.0, 0.0))
        self.orientation = Matrix44.identity()

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


cam = Camera()

h = 1080
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
    Vector4((1.0, 1.0, 0.0, 1.0)),
    Vector3((0.0, 0.0, 0.0)),
    0.0,
)
material2 = Material(
    Vector4((0.6, 0.3, 1.0, 1.0)),
    Vector3((0.0, 0.0, 0.0)),
    0.0,
)
# light source
material3 = Material(
    Vector4((0.5, 0.3, 0.8, 1.0)),
    Vector3((0.5, 0.3, 0.8)),
    0.3,
)
material4 = Material(
    Vector4((0.1, 0.7, 0.3, 1.0)),
    Vector3((0.0, 0.0, 0.0)),
    0.0,
)


spheres = [
    Sphere(
        pos=Vector3((0.0, -1001.0, 8)),
        radius=1000.0,
        material=material4,
    ),
    Sphere(
        pos=Vector3((0.0, 0.0, 10.0)),
        radius=1.0,
        material=material1,
    ),
    Sphere(
        pos=Vector3((2.0, 0.0, 10.0)),
        radius=0.5,
        material=material2,
    ),
    Sphere(
        pos=Vector3((-3.0, 0.0, 10.0)),
        radius=0.9,
        material=material2,
    ),
    Sphere(
        pos=Vector3((-2.0, 0.0, 7.0)),
        radius=0.6,
        material=material3,
    ),
    # Sphere(
    #     pos=Vector3((0.0, 13, 10)),
    #     radius=0.8,
    #     material=material3,
    # ),
    # Sphere(
    #     pos=Vector3((0.0, 7, 10)),
    #     radius=0.6,
    #     material=material2,
    # ),
]

buffer1 = ctx.buffer(vertices.tobytes())

vao = ctx.vertex_array(
    program,
    [
        (buffer1, "3f", "position"),
    ],
)


fbo = ctx.framebuffer(color_attachments=[ctx.texture((w, h), 3)])
fbo.clear()

# initialise the uniforms

cam.orientation = Matrix44.identity()

program["screenWidth"].write(struct.pack("i", w))
program["screenHeight"].write(struct.pack("i", h))

program["ViewParams"].write(cam.view_params.astype("f4"))
program["CamLocalToWorldMatrix"].write(cam.local_to_world_matrix.astype("f4"))
program["CamGlobalPos"].write(cam.pos.astype("f4"))

sphere_buffer_binding = 1
program["sphereBuffer"].binding = sphere_buffer_binding

# program["frameBuffer"]


running = True
try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        time = pygame.time.get_ticks() / np.float32(1000.0)
        # time = 0.0
        # program["time"].write(time.astype("f4"))

        # render_object.render(mode=moderngl.TRIANGLE_STRIP)

        program["spheresCount"].write(struct.pack("i", len(spheres)))

        spheres[4].pos.x = 2 * sin(time * 2.1)
        spheres[4].pos.y = 5 + 3 * cos(time * 1.9)
        spheres[4].pos.z = 8 + 5 * cos(time * 1.1)

        # spheres[1].pos.x = 1.5 * sin(time * 2)
        # spheres[1].pos.y = 0.5 * cos(time * 6)
        # spheres[1].pos.z = 12 + 8 * cos(time * 1)

        # sphere_bytes = struct.pack("<i", len(spheres))
        sphere_bytes = b""
        for sphere in spheres:
            sphere_bytes += sphere.tobytes()
        sphere_bytes += b"\x00" * (16 - len(sphere_bytes) % 16)

        # sphere_bytes = struct.pack("i", len(spheres)) + iter_to_bytes(spheres)
        # sphere_bytes = iter_to_bytes(spheres)
        sphere_buffer = ctx.buffer(sphere_bytes)
        sphere_buffer_binding = 1
        sphere_buffer.bind_to_uniform_block(sphere_buffer_binding)

        vao.render(mode=moderngl.TRIANGLE_STRIP)

        pygame.display.flip()
        clock.tick(24)

        pass

except KeyboardInterrupt:
    pass
