from __future__ import annotations


import numpy as np
from typing import Iterable

from PIL import Image

from .classes import (
    Ray,
    HitInfo,
    ByteableObject,
    Triangle,
    Scene,
    Camera,
    Csys,
)

import pyrr
from pyrr import Vector3, Vector4, Matrix33, Matrix44


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def raySphere(ray: Ray, spherePos: np.ndarray, rad: float) -> HitInfo:
    """A function that calculates the intersection of a ray and sphere"""
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


def iter_to_bytes(iterable: Iterable[ByteableObject]) -> bytearray:
    """Converts an iterable byteable object to bytes using the ByteableObject protocol"""
    ret = bytearray()
    for x in iterable:
        ret.extend(x.tobytes())
    return ret


def buffer_to_image_float16(
    buffer: bytes, size: tuple[int, int], mode="RGB"
) -> Image.Image:
    """Reads a buffer image into float 16"""
    buffer_np = np.frombuffer(buffer, np.float16).reshape((size[1], size[0], len(mode)))
    buffer_np = np.flipud(buffer_np)
    img = Image.fromarray(buffer_np, mode)
    return img


def buffer_to_image(buffer: bytes, size: tuple[int, int], mode="RGB") -> Image.Image:
    """Reads a buffer directly to image"""
    img = Image.frombuffer(mode, size, buffer)
    # need to flip the image due to the read direction of the buffer.
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def rayTriangle(
    ray: Ray,
    tri: Triangle,
) -> HitInfo:
    """A function that calculates the intersection between a ray and a triangle"""

    # Refer to the shader code for details

    return HitInfo()

    hitInfo = HitInfo()

    v0 = tri.posA
    v1 = tri.posB
    v2 = tri.posC

    dir = ray.dir
    orig = ray.origin

    v0v1 = v1 - v0
    v0v2 = v2 - v0

    N = v0v1.cross(v0v2)

    kEpsilon = 1e-6
    NDotRayDirection = N.dot(dir)
    if abs(NDotRayDirection) < kEpsilon:
        hitInfo.didHit = False
        return hitInfo

    d = -N.dot(v0)

    t = -(N.dot(orig) + d) / NDotRayDirection

    if t < 0.0:
        hitInfo.didHit = False
        return hitInfo

    P = orig + t * dir

    C: Vector3

    edge0 = v1 - v0
    vp0 = P - v0
    C = edge0.cross(vp0)
    if N.dot(C) < 0.0:
        hitInfo.didHit = False
        return hitInfo

    edge1 = v2 - v1
    vp1 = P - v1
    C = edge1.cross(vp1)
    if N.dot(C) < 0.0:
        hitInfo.didHit = False
        return hitInfo

    edge2 = v0 - v2
    vp2 = P - v2
    C = edge2.cross(vp2)
    if N.dot(C) < 0.0:
        hitInfo.didHit = False
        return hitInfo

    hitInfo.normal = normalize(N)
    hitInfo.didHit = True
    hitInfo.dst = t
    hitInfo.hitPoint = ray.origin + hitInfo.dst * ray.dir

    return hitInfo


def boundingBoxIntersect(
    ray: Ray,
    bboxMin: Vector3,
    bboxMax: Vector3,
) -> bool:

    # // calculates the boolean intersection between a aabb and a ray
    tmin = (bboxMin.x - ray.origin.x) / ray.dir.x
    tmax = (bboxMax.x - ray.origin.x) / ray.dir.x

    # if (tmin > tmax) swap(tmin, tmax)
    t1, t2 = tmin, tmax
    tmin, tmax = min(t1, t2), max(t1, t2)

    tymin = (bboxMin.y - ray.origin.y) / ray.dir.y
    tymax = (bboxMax.y - ray.origin.y) / ray.dir.y

    # if (tymin > tymax) swap(tymin, tymax)
    t1, t2 = tymin, tymax
    tymin, tymax = min(t1, t2), max(t1, t2)

    if (tmin > tymax) or (tymin > tmax):
        return False

    if tymin > tmin:
        tmin = tymin

    if tymax < tmax:
        tmax = tymax

    tzmin = (bboxMin.z - ray.origin.z) / ray.dir.z
    tzmax = (bboxMax.z - ray.origin.z) / ray.dir.z

    # if (tzmin > tzmax) swap(tzmin, tzmax)
    t1, t2 = tzmin, tzmax
    tzmin, tzmax = min(t1, t2), max(t1, t2)

    if (tmin > tzmax) or (tzmin > tmax):
        return False

    if tzmin > tmin:
        tmin = tzmin
    if tzmax < tmax:
        tmax = tzmax

    return True


def rayScene(
    ray: Ray,
    scene: Scene,
) -> list:
    """Traces a non-reflecting ray through the scene using the camera
    and determines the intersecting objects and returns a list of hit
    infos, the objects hit, and the distances/positions
    """

    spheres = scene.spheres
    meshes = scene.meshes

    ret = []

    for sphere in spheres:

        hit_info = raySphere(ray, sphere.pos, sphere.radius)

        if not hit_info.didHit:
            continue

        # hit_position = ray.dir * hit_info.dst + ray.origin
        hit_position = hit_info.hitPoint

        ret.append((hit_info.dst, sphere, hit_position))

    for mesh in meshes:
        bbox = mesh.bounding_box

        hits_bbox = boundingBoxIntersect(ray, *bbox)

        # ignore the meshes we aren't intersecting bboxes with
        if not hits_bbox:
            continue

        # loop over all of the triangles in that mesh to find the interesections
        for triangle in mesh.triangles:
            hit_info = rayTriangle(ray, triangle)
            if not hit_info.didHit:
                continue

            # hit_position = ray.dir * hit_info.dst + ray.origin
            hit_position = hit_info.hitPoint

            ret.append((hit_info.dst, triangle, hit_position))

    # sort the objects by the shortest distance
    ret.sort(key=lambda x: x[0])

    return ret


def convert_screen_coords_to_camera_ray(
    x0: float,
    y0: float,
    window_width: float,
    window_height: float,
    cam: Camera,
) -> Ray:

    x1 = Vector4(
        (
            (x0 - window_width / 2) / (window_width) * 1,
            -(y0 - window_height / 2) / (window_height) * 1,
            1.0,
            1.0,
        )
    )

    x2 = x1 * Vector4((*cam.view_params, 1.0))

    x3: Vector4 = cam.local_to_world_matrix.transpose() * x2

    ray = Ray()
    direction = Vector3(x3.xyz)
    direction.normalise()
    ray.dir = direction
    ray.origin = cam.csys.pos

    return ray


pass
