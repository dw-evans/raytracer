from __future__ import annotations


import numpy as np
from typing import Iterable

from PIL import Image

from .classes import (
    Ray,
    HitInfo,
    ByteableObject,
)


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


def iter_to_bytes(iterable: Iterable[ByteableObject]) -> bytearray:
    ret = bytearray()
    for x in iterable:
        ret.extend(x.tobytes())
    return ret


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
