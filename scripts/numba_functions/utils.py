from __future__ import annotations

import pyrr
from numba import jit, vectorize, cuda
from numba.types import float32
from numba.experimental import jitclass

import numpy as np
import time
from typing import Callable
from functools import wraps

pyrr.quaternion.create()


def timer(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t_start = time.time()
        ret = fn(*args, **kwargs)
        t_end = time.time()
        print(f"Function {fn.__qualname__} executed in {(t_end - t_start) * 1e3} ms")
        return ret

    return wrapper


@jit(nopython=True)
# @cuda.jit
def normalize(vec: np.ndarray):
    return vec / np.linalg.norm(vec)


@jit(nopython=True)
# @cuda.jit
def mat33_create_from_quaternion(quat: np.ndarray):
    """Creates a matrix with the same rotation as a quaternion.

    :param quat: The quaternion to create the matrix from.
    :rtype: numpy.array
    :return: A matrix with shape (3,3) with the quaternion's rotation.
    """

    dtype = quat.dtype

    # the quaternion must be normalized
    if not np.isclose(np.linalg.norm(quat), 1.0):
        quat = normalize(quat)

    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]

    sqw = qw**2
    sqx = qx**2
    sqy = qy**2
    sqz = qz**2
    qxy = qx * qy
    qzw = qz * qw
    qxz = qx * qz
    qyw = qy * qw
    qyz = qy * qz
    qxw = qx * qw

    invs = 1 / (sqx + sqy + sqz + sqw)
    m00 = (sqx - sqy - sqz + sqw) * invs
    m11 = (-sqx + sqy - sqz + sqw) * invs
    m22 = (-sqx - sqy + sqz + sqw) * invs
    m10 = 2.0 * (qxy + qzw) * invs
    m01 = 2.0 * (qxy - qzw) * invs
    m20 = 2.0 * (qxz - qyw) * invs
    m02 = 2.0 * (qxz + qyw) * invs
    m21 = 2.0 * (qyz + qxw) * invs
    m12 = 2.0 * (qyz - qxw) * invs

    return np.array(
        [
            [m00, m01, m02],
            [m10, m11, m12],
            [m20, m21, m22],
        ],
        dtype=quat.dtype,
    )

@jit(nopython=True)
def mat44_create_from_quaternion(quat: np.ndarray):
    dtype = quat.dtype
    ret = np.eye(4, 4)
    ret[0:3, 0:3] = mat33_create_from_quaternion(quat=quat)
    return ret

pyrr.Matrix44.inverse

@jit(nopython=True)
# @cuda.jit
def quaternion_create_from_axis_rotation(axis: np.ndarray, theta):
    dtype = axis.dtype
    # make sure the vector is normalized
    if not np.isclose(np.linalg.norm(axis), 1.0):
        axis = normalize(axis)

    thetaOver2 = theta * 0.5
    sinThetaOver2 = np.sin(thetaOver2)

    return np.array(
        [
            sinThetaOver2 * axis[0],
            sinThetaOver2 * axis[1],
            sinThetaOver2 * axis[2],
            np.cos(thetaOver2),
        ],
        dtype=dtype,
    )


spec_csys = [
    ("pos", float32[:]),
    ("quat", float32[:]),
]


@jitclass(spec=spec_csys)
class NumbaCsys:
    def __init__(self) -> None:
        self.pos = np.array([0, 0, 0], dtype=np.float32)
        self.quat = np.array([0, 0, 0, 1], dtype=np.float32)

    @property
    def rotation_matrix(self):
        return mat33_create_from_quaternion(self.quat)

    @property
    def transformation_matrix(self):
        ret = mat44_create_from_quaternion(self.quat)
        ret[3][0:3] = self.pos
        return ret


    def rzg(self, degrees: float) -> NumbaCsys:
        rad = np.radians(degrees)
        ax = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.quat = self.quat * quaternion_create_from_axis_rotation(ax, rad)
        return self


spec_triangle = [
    ("posA", float32[:]),
    ("posB", float32[:]),
    ("posC", float32[:]),
    ("normalA", float32[:]),
    ("normalB", float32[:]),
    ("normalC", float32[:]),
]


def obj_import_testing():
    import trimesh

    from scripts.classes import Mesh, Material, Triangle


    @timer
    def from_obj_original(
        vertex_indices: np.ndarray, vertices: np.ndarray, vertex_normals: np.ndarray
    ):

        ret = Mesh(material=Material())
        tris = []
        total_faces = len(vertex_indices)
        for i, face in enumerate(vertex_indices):
            vertex_indices = face
            v0, v1, v2 = vertices[vertex_indices]
            n0, n1, n2 = vertex_normals[vertex_indices]
            tris.append(
                Triangle(
                    posA=v0,
                    posB=v1,
                    posC=v2,
                    normalA=n0,
                    normalB=n1,
                    normalC=n2,
                    parent=None,
                )
            )
            if (i % 100) == 0:
                completion = i / total_faces
                print(f"  {completion * 100:>6.4f}%", end="\r")

        ret.add_tri(tris)
        print("Loading complete.")
        return ret


    @timer
    @jit(nopython=True)
    def from_obj_optimized(
        vertex_indices_arr: np.ndarray, vertices: np.ndarray, vertex_normals: np.ndarray
    ):

        n = len(vertex_indices_arr)
        ret = np.zeros((n, 18), dtype=np.float32)
        for i, vertex_indices in enumerate(vertex_indices_arr):

            ret[i][0:9] = vertices[vertex_indices].flatten()
            ret[i][9:18] = vertex_normals[vertex_indices].flatten()

            # if (i % 100) == 0:
            # completion = i / n
            # print("  {}%".format(completion * 100), end="\r")

        return ret


    msh = trimesh.load("objects/tyre_tread.obj")

    vertex_indices_arr = msh.faces.astype(np.int32)
    vertices = msh.vertices.astype(np.float32)
    vertex_normals = msh.vertex_normals.astype(np.float32)


    print("Including Compilation:")
    from_obj_optimized(vertex_indices_arr, vertices, vertex_normals)
    from_obj_original(vertex_indices_arr, vertices, vertex_normals)

    print("")
    print("Excluding Compilation:")
    from_obj_optimized(vertex_indices_arr, vertices, vertex_normals)


def csys_comparison_testing():
    COUNT = 1_000_000
    @timer
    @jit(nopython=True)
    def operate_on_csys1(csys:NumbaCsys, count=1):
        x = np.zeros(4)
        for i in COUNT:
            x += csys.transformation_matrix
            csys.rzg(1.0)

    from scripts.classes import Csys

    @timer
    def operate_on_csys2(csys:Csys, count=1):
        x = np.zeros(4)
        for i in COUNT:
            x += csys.transformation_matrix
            csys.rzg(1.0)

    x1 = NumbaCsys()
    x2 = Csys()

    print("Including Compilation:")
    operate_on_csys1(x1)
    # operate_on_csys2(x2)

    print("")
    print("Excluding Compilation:")
    operate_on_csys1(x1, COUNT)
    operate_on_csys2(x2, COUNT)

    pass

csys_comparison_testing()

pass



