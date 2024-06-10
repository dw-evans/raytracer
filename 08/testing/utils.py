from __future__ import annotations

import moderngl
import pygame
import numpy as np


def basic_shader_program(
    w: int,
    h: int,
    vertex_shader: str,
    fragment_shader: str,
) -> tuple[
    moderngl.Context,  # ctx
    moderngl.Program,  # program
    moderngl.VertexArray,  # vao
    pygame.time.Clock,
]:
    pygame.init()

    screen = pygame.display.set_mode(
        (w, h),
        pygame.OPENGL | pygame.DOUBLEBUF,
    )

    clock = pygame.time.Clock()

    ctx = moderngl.create_context()

    program = ctx.program(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader,
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

    buffer1 = ctx.buffer(vertices.tobytes())

    vao = ctx.vertex_array(
        program,
        [
            (buffer1, "3f", "position"),
        ],
    )

    return (
        ctx,
        program,
        vao,
        clock,
    )
