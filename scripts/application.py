from __future__ import annotations


import pygame
import moderngl

from typing import Protocol
from abc import ABC
import struct

import functions

import classes
from classes import (
    Scene,
    Sphere,
    Triangle,
    Mesh,
    Camera,
    Csys,
)

from pyrr import (
    Vector3,
    Vector4,
    Matrix33,
    Matrix44,
)

from constants import *


class ShaderProgram:
    DUMB = "dumb"
    RAYTRACER = "raytracer"
    SILHOUETTE = "silhouette"


class Application:
    context: moderngl.Context
    active_program: moderngl.Program
    programs: list[moderngl.Program]
    scene: Scene

    def __init__(
        self,
    ) -> None:
        pass

    def configure(self) -> None:
        pass

    def render(self) -> None:
        pass

    def register_program(self, program: Program):
        if not program in self.programs:
            self.programs.append(program)
        else:
            print("Warning, program already regiested")


class Program(Protocol):
    prog: moderngl.Program

    def __init__(
        self,
        vertex_shader_file: str,
        fragment_shader_file: str,
    ) -> None:
        super().__init__()

        with open(vertex_shader_file) as f:
            shader_fragment = f.read()

        with open(fragment_shader_file) as f:
            shader_vertex = f.read()

        self.prog = Application.context.program(
            vertex_shader=shader_vertex,
            fragment_shader=shader_fragment,
        )

        Application.register_program(self)

    def calculate(
        self,
        ctx: moderngl.Context,
        prog: moderngl.Program,
    ):
        """What is called during the main application loop"""
        raise NotImplementedError

    def configure(
        self,
        ctx: moderngl.Context,
        prog: moderngl.Program,
    ):
        """What is run to configure the program"""
        raise NotImplementedError


class RayTracingProgram(Program):

    scene = Application.scene

    def __init__(self) -> None:
        super().__init__(
            fragment_shader_file=f"scripts/{ShaderProgram.RAYTRACER}.fs.glsl",
            vertex_shader_file=f"scripts/{ShaderProgram.RAYTRACER}.vs.glsl",
        )

    def calculate(self, ctx: moderngl.Context, prog: moderngl.Program):
        return super().calculate(ctx, prog)

    def configure(self, ctx: moderngl.Context, prog: moderngl.Program):

        w, h = WINDOW_WIDTH, WINDOW_HEIGHT

        spheres = self.scene.spheres
        triangles = self.scene.triangles
        meshes = self.scene.meshes

        prog["STATIC_RENDER"].write(struct.pack("i", STATIC_RENDER))
        prog["MAX_BOUNCES"].write(struct.pack("i", MAX_RAY_BOUNCES))
        prog["RAYS_PER_PIXEL"].write(struct.pack("i", RAYS_PER_PIXEL))

        texture = ctx.texture((w, h), 3)
        texture.use(location=1)
        prog["previousFrame"] = 1
        render_data = b"\x00" * w * h * 3

        # initialise the uniforms
        prog["screenWidth"].write(struct.pack("i", w))
        prog["screenHeight"].write(struct.pack("i", h))

        prog["spheresCount"].write(struct.pack("i", len(self.scene.spheres)))
        sphere_buffer_binding = 1
        prog["sphereBuffer"].binding = sphere_buffer_binding

        triangles = self.scene.triangles
        n_triangles = len(triangles)
        prog["triCount"].write(struct.pack("i", n_triangles))

        prog["meshCount"].write(struct.pack("i", len(meshes)))

        triangles_ssbo = ctx.buffer(
            functions.iter_to_bytes(
                [t.update_pos_with_mesh2() for t in triangles],
            )
        )
        triangles_ssbo_binding = 9
        triangles_ssbo.bind_to_storage_buffer(binding=triangles_ssbo_binding)

        mesh_buffer = ctx.buffer(functions.iter_to_bytes(meshes))
        mesh_buffer_binding = 10
        prog["meshBuffer"].binding = mesh_buffer_binding
        mesh_buffer.bind_to_uniform_block(mesh_buffer_binding)

        sky_color = Vector3((131, 200, 228), dtype="f4") / 255
        ground_color = Vector3((74, 112, 45), dtype="f4") / 255

        prog["skyColor"].write(struct.pack("3f", *sky_color))
