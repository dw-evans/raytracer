# Testing out the insideoutise function
from pyrr import (
    Vector3,
    Vector4,
    Matrix44,
)

import pygame
import numpy as np
from numpy import tan, radians, sin, cos

from typing import Iterable


class Ray:
    def __init__(
        self,
        origin: Vector3 = Vector3((0.0, 0.0, 0.0)),
        dir: Vector3 = Vector3((0.0, 0.0, 0.0)),
    ) -> None:
        self.origin = origin
        self.dir = dir


class Triangle:
    def __init__(
        self,
        v0: Vector3,
        v1: Vector3,
        v2: Vector3,
    ) -> None:
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2

    @property
    def normal(self):
        edge_ab = self.v1 - self.v0
        edge_ac = self.v2 - self.v0
        return edge_ab.cross(edge_ac)


class HitInfo:
    def __init__(self) -> None:
        self.didHit: bool = False
        self.dst: float = 0.0
        self.hitPoint: Vector3 = Vector3((0.0, 0.0, 0.0))
        self.normal: Vector3 = Vector3((0.0, 0.0, 0.0))


class Camera:
    def __init__(self) -> None:
        self.fov = 28.072  # degrees
        self.aspect = 16.0 / 9.0
        self.near_plane = 240.0
        self.pos = Vector3((0.0, 0.0, 0.0))
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        # self.fov = 30.0  # degrees
        # self.aspect = 16.0 / 9.0
        # self.near_plane = 240.0
        # self.pos = Vector3((0.0, 0.0, 0.0))
        # self.roll = 0.0
        # self.pitch = 0.0
        # self.yaw = 0.0

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


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def rayTriangleIntersect(ray: Ray, tri: Triangle) -> HitInfo:

    hit = HitInfo()

    v0, v1, v2 = tri.v0, tri.v1, tri.v2
    orig, dir = ray.origin, ray.dir

    v0v1 = v1 - v0
    v0v2 = v2 - v0

    N = v0v1.cross(v0v2)

    area2 = N.length

    #  finding P
    kEpsilon = 1e-6
    NdotRayDirection = N.dot(dir)
    if abs(NdotRayDirection) < kEpsilon:
        hit.didHit = False
        return hit

    # computing d
    d = -N.dot(v0)

    # compute t
    t = -(N.dot(orig) + d) / NdotRayDirection

    # if the triangle is behind the ray no hit
    if t < 0.0:
        hit.didHit = False
        return hit

    # compute the intersction point of the plan
    P = orig + t * dir

    # check the if the intersection point is within the triangle for all three edges
    # edge 0
    edge0 = v1 - v0
    vp0 = P - v0
    C = edge0.cross(vp0)
    if N.dot(C) < 0.0:
        hit
        hit.didHit = False
        return hit

    # edge 1
    edge1 = v2 - v1
    vp1 = P - v1
    C = edge1.cross(vp1)
    if N.dot(C) < 0.0:
        hit.didHit = False
        return hit

    # edge 2
    edge2 = v0 - v2
    vp2 = P - v2
    C = edge2.cross(vp2)
    if N.dot(C) < 0.0:
        hit.didHit = False
        return hit

    hit.didHit = True
    hit.normal = tri.normal

    return hit


def main() -> None:

    cam = Camera()

    h = 30

    # I think the near plane needs to be scaled with the height to make it work
    cam.fov = 28.072
    cam.near_plane = cam.near_plane / (120 / h)

    # cam.pos.z = 30
    # cam.yaw = 180

    w = int(h * cam.aspect)
    scale_factor = 4
    pygame.init()

    # ray = Ray(Vector3((0.0, 0.0, 0.0)), Vector3(0., 0., 1.))

    # v0 = Vector3((-2.0, -2.0, 15.0))
    # v1 = Vector3((2.0, -2.0, 15.0))
    # v2 = Vector3((0.0, 4.0, 15.0))

    v0 = Vector3((0.0, 3.0, 15.0))
    v1 = Vector3((-1.0, 0.0, 15.0))
    v2 = Vector3((2.0, 0.0, 15.0))

    tri = Triangle(v0, v1, v2)

    pygame.init()
    screen = pygame.display.set_mode(
        (w * scale_factor, h * scale_factor),
        pygame.DOUBLEBUF,
    )

    clock = pygame.time.Clock()

    rays = np.ndarray((w, h), dtype="object")

    for i in range(w):
        for j in range(h):
            r = Ray()
            r.origin = Vector3((0.0, 0.0, 0.0))
            r.dir = Vector3(
                (i - cam.plane_width / 2, j - cam.plane_height / 2, cam.near_plane)
            )
            r.dir.normalize()
            rays[i, j] = r

    # ray = rays[rays.shape[0] // 2, rays.shape[1] // 2]
    # hit = rayTriangleIntersect(ray, tri)

    # pass

    running = True
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
                    rect = pygame.Rect(
                        i * scale_factor, j * scale_factor, scale_factor, scale_factor
                    )
                    ray: Ray = rays[i, j]

                    pos = Vector3(
                        [
                            (i - cam.plane_width / 2) / cam.plane_width,
                            (j - cam.plane_height / 2) / cam.plane_height,
                            1.0,
                        ]
                    )

                    viewPointLocal = Vector3(
                        [
                            pos.x * cam.plane_width,
                            pos.y * cam.plane_height,
                            pos.z * cam.near_plane,
                        ]
                    )

                    viewPoint = cam.local_to_world_matrix * cam.view_params
                    viewPoint = Vector3(viewPointLocal.xyz)

                    r.origin

                    x = viewPoint - ray.origin
                    x.normalise()

                    hit = rayTriangleIntersect(ray, tri)

                    if hit.didHit:
                        pass

                    # color = (
                    #     abs(ray.dir.x) * 255,
                    #     abs(ray.dir.y) * 255,
                    #     abs(ray.dir.z) * 255,
                    # )
                    color = (
                        255 if hit.didHit else 0.0,
                        255 if hit.didHit else 0.0,
                        255 if hit.didHit else 0.0,
                    )
                    # color = (
                    #     abs(hit.normal[0]) * 255,
                    #     abs(hit.normal[1]) * 255,
                    #     abs(hit.normal[2]) * 255,
                    # )

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
            clock.tick(24)

    except KeyboardInterrupt:
        pass

    # pass

    pass


if __name__ == "__main__":
    main()
