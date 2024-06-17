import struct

import moderngl

import pygame
import sys

pygame.init()


wnd = pygame.display.set_mode(
    (600, 480),
    pygame.OPENGL | pygame.DOUBLEBUF,
)

clock = pygame.time.Clock()


ctx = moderngl.create_context()

# Shaders & Program

prog = ctx.program(
    vertex_shader=(
        """
        #version 330

        in vec2 vert;

        in vec4 vert_color;
        out vec4 frag_color;

        uniform vec2 scale;
        uniform float rotation;

        void main() {
            frag_color = vert_color;
            mat2 rot = mat2(
                cos(rotation), sin(rotation),
                -sin(rotation), cos(rotation)
            );
            gl_Position = vec4((rot * vert) * scale, 0.0, 1.0);
        }
    """
    ),
    fragment_shader=(
        """
        #version 330

        in vec4 frag_color;
        out vec4 color;

        void main() {
            color = vec4(frag_color);
        }
    """
    ),
)


scale = prog["scale"]
rotation = prog["rotation"]

width, height = wnd.get_size()
scale.value = (height / width * 0.75, 0.75)

# Buffer

vbo = ctx.buffer(
    struct.pack(
        "18f",
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.5,
        -0.5,
        0.86,
        0.0,
        1.0,
        0.0,
        0.5,
        -0.5,
        -0.86,
        0.0,
        0.0,
        1.0,
        0.5,
    )
)

# Put everything together

# vao = ctx.simple_vertex_array(prog, vbo, ["vert", "vert_color"])

vao = ctx.vertex_array(
    prog,
    [
        (vbo, "2f", "vert"),
        (vbo, "4f", "vert_color"),
    ],
)


# Main loop

try:
    while True:
        ctx.clear(0.9, 0.9, 0.9)
        ctx.enable(moderngl.BLEND)
        rotation.value = pygame.time.get_ticks() / 1000.0
        vao.render(instances=10)

        clock.tick(24)
except KeyboardInterrupt:
    sys.exit()
