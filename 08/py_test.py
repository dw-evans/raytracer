import pygame
import moderngl
import os
import sys

import numpy as np
from numpy import sin, cos, tan, radians
from array import array
from pathlib import Path

from pyrr import Vector3, Matrix44, Vector4

os.chdir(Path(__file__).parent)
print(os.getcwd())


class HitInfo:
    def __init__(self) -> None:
        self.didHit:bool = False
        self.dst:float = np.inf
        self.hitPoint = np.array([0.,0.,0.])
        self.normal = np.array([0.,0.,0.])

class Ray:
    def __init__(self) -> None:
        self.origin = np.array([0., 0., 0.])
        self.dir = np.array([0., 0., 0.])


# vec3 = np.ndarray[float, float, float]

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def raySphere(ray:Ray, spherePos:np.ndarray, rad:float) -> HitInfo:

    ret = HitInfo()

    offsetRayOrigin = ray.origin - spherePos

    a = np.dot(ray.dir, ray.dir)
    b = 2 * np.dot(offsetRayOrigin, ray.dir)
    c = np.dot(offsetRayOrigin, offsetRayOrigin) - rad * rad

    discriminant = b * b - 4 * a * c

    if discriminant >= 0:
        dst = (-b - np.sqrt(discriminant)) / (2*a)

        if dst >= 0:
            ret.didHit = True
            ret.dst = dst
            ret.hitPoint = ray.origin + ray.dir * dst
            ret.normal = normalize(ret.hitPoint - spherePos)


    return ret

USE_SHADERS = True

cam_aspect = 16.0 / 9.0

h = 120
w = int(h * cam_aspect)
SCALE_FACTOR = 3
pygame.init()

screen = pygame.display.set_mode(
    (w * SCALE_FACTOR, h * SCALE_FACTOR),
    pygame.OPENGL | pygame.DOUBLEBUF,
)

clock = pygame.time.Clock()

ctx = moderngl.create_context()

vertices = ctx.buffer(
    array('f', [
        -1.0, 1.0, 0.,
        1.0, 1.0, 0.,
        -1.0, -1.0, 0.,
        1.0, -1.0, 0.,
    ])
)


# get the vertex and fragment shaders
shader_vertex = ""
shader_fragment = ""

with open("shaders_v2/fs.glsl") as f:
    shader_fragment = f.read()

with open("shaders_v2/vs.glsl") as f:
    shader_vertex = f.read()


program = ctx.program(
    vertex_shader=shader_vertex,
    fragment_shader=shader_fragment,
)

# vbo = ctx.buffer(vertices.tobytes())

render_object = ctx.vertex_array(
    program,
    [(vertices, "3f", "position")]
)

# render_object = ctx.vertex_array(
#     program,
#     vbo,
#     "position",
# )


# initialise the uniforms

cam_fov = 28.072
cam_near_plane = 240.0

cam_orientation = Matrix44.identity()

plane_height = cam_near_plane * tan(radians(cam_fov) * 0.5) * 2.0
plane_width = plane_height * cam_aspect

view_params = Vector3(
    [plane_width, plane_height, cam_near_plane],
)

cam_global_pos = Vector3(
    [0.0, 0.0, 0.0],
)

cam_local_to_world_matrix = Matrix44.identity()

if USE_SHADERS:

    program["ViewParams"].write(view_params.astype("f4"))
    program["CamLocalToWorldMatrix"].write(cam_local_to_world_matrix.astype("f4"))
    program["CamGlobalPos"].write(cam_global_pos.astype("f4"))

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()

            time = pygame.time.get_ticks() / np.float32(1000.0)
            program["time"].write(time.astype("f4"))

            render_object.render(mode=moderngl.TRIANGLE_STRIP)

            pygame.display.flip()
            clock.tick(60)

    except KeyboardInterrupt:
        pass
        
# ray = Ray()
# ray.origin = np.array([0,0,0])
# ray.dir = np.array([0.0,0.09,1])

# spherePos = np.array([0.0,0,10])
# rad = 1.0

# res = raySphere(ray, spherePos, rad)
# print(f"ray.origin={ray.origin}\nray.dir={ray.dir}\n")
# print(f"res.didHit={res.didHit}\nres.dst={res.dst}\nres.hitPoint={res.hitPoint}\nres.normal={res.normal}\n\n")


if not USE_SHADERS:


    pygame.init()

    screen = pygame.display.set_mode(
        (w * SCALE_FACTOR, h * SCALE_FACTOR),
        pygame.DOUBLEBUF,
    )

    clock = pygame.time.Clock()


    rays = np.ndarray((w, h),dtype="object")

    for i in range(w):
        for j in range(h):
            r = Ray()
            r.origin = np.array([0,0,0])
            r.dir = normalize([i-plane_width / 2, j-plane_height / 2, cam_near_plane])
            rays[i,j] = r
            # rays[i,j] = np.array([0,0,0]) - np.array([])


    running = True

    """
    vec3 viewPointLocal = (vec3(position.xy - 0.5, 1) * ViewParams);
    vec3 viewPoint = (CamLocalToWorldMatrix * vec4(viewPointLocal.xyz, 1.0)).xyz;

    Ray ray;
    ray.origin = CamGlobalPos;
    ray.dir = normalize(viewPoint - ray.origin);
    """

    try:
        while running:
            for event in pygame.event.get():
                # only do something if the event is of type QUIT
                if event.type == pygame.QUIT:
                    # change the value to False, to exit the main loop
                    running = False

            screen.fill((10, 10, 10))

            for i in range(w):
                for j in range(h):
                    rect = pygame.Rect(i*SCALE_FACTOR, j*SCALE_FACTOR, SCALE_FACTOR, SCALE_FACTOR)

                    ray:Ray = rays[i, j]
                    hit = raySphere(ray, np.array([0.0, 0.0, 5.0]), 1.0)

                    pos = Vector3([
                        (i - plane_width / 2) / plane_width, 
                        (j - plane_height / 2) / plane_height, 
                        1.0,
                    ])

                    viewPointLocal = Vector3([
                        pos.x * plane_width, 
                        pos.y * plane_height, 
                        pos.z * cam_near_plane,
                    ])

                    viewPoint = cam_local_to_world_matrix * Vector4(
                        [viewPointLocal.x, viewPointLocal.y, viewPointLocal.z, 1.0]
                    )
                    viewPoint = Vector3(viewPointLocal.xyz)

                    r.origin

                    x = viewPoint - ray.origin
                    x.normalise()

                    if hit.didHit:
                        pass


                    color=(
                        abs(hit.normal[0])*255,
                        abs(hit.normal[1])*255,
                        abs(hit.normal[2])*255,
                    )

                    # color=(
                    #     max(hit.normal[0], 0)*255,
                    #     max(hit.normal[1], 0)*255,
                    #     max(hit.normal[2], 0)*255,
                    # )

                    # color = np.maximum(np.minimum(ray.dir * 255 * 1e6, 255), 0)

                    pygame.draw.rect(
                        surface=screen, 
                        color=color,
                        rect=rect,
                    )



            pygame.display.flip()
            clock.tick(1)
            

    except KeyboardInterrupt:
        pass


# pass