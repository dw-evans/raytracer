from __future__ import annotations

import moderngl
import pygame
import numpy as np
import sys
import struct

import utils

vertex_shader = """
#version 330 core

layout (location = 0) in vec3 position;
out vec3 fragPosition;

void main() {
    gl_Position = vec4(position.x,  position.y, position.z, 1.0);
    fragPosition = gl_Position.xyz;
}
"""

fragment_shader = """
#version 330 core

in vec3 fragPosition;
out vec4 color;

uniform uint screenWidth;
uniform uint screenHeight;

float randomValue(inout uint state) 
{
    state *= (state + uint(195439)) * (state + uint(124395)) * (state + uint(845921));
    return state / 4294967295.0;
}


void main() 
{
    uint numPixels = screenWidth * screenHeight;
    vec4 pxCoord = gl_FragCoord;
    uint pxId = uint(pxCoord.x * screenWidth * screenHeight) + uint(pxCoord.y * screenHeight);
    
    uint rngState = pxId;

    color = vec4(
        randomValue(rngState),
        randomValue(rngState),
        randomValue(rngState),
        1.0
    );
}

"""
w, h = 360, 180

scale_factor = 4

(
    ctx,
    program,
    vao,
    clock,
) = utils.basic_shader_program(
    w,
    h,
    vertex_shader,
    fragment_shader,
)


running = True

program["screenWidth"].write(struct.pack("i", w))
program["screenHeight"].write(struct.pack("i", h))

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        time = pygame.time.get_ticks() / np.float32(1000.0)

        vao.render(mode=moderngl.TRIANGLE_STRIP)

        pygame.display.flip()
        clock.tick(144)

except KeyboardInterrupt:
    pass
