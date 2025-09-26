from __future__ import annotations

import pyrr
from numba import jit
from numba.types import float32, int32
from numba.experimental import jitclass

import numpy as np
import time
from typing import Callable
from functools import wraps


def timer(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t_start = time.time()
        ret = fn(*args, **kwargs)
        t_end = time.time()
        print(f"Function {fn.__name__} executed in {(t_end - t_start) * 1e3:.2f} ms")
        return ret

    return wrapper


@jit(nopython=True, cache=True)
def normalize(vec: np.ndarray):
    return vec / np.linalg.norm(vec)

@jit(nopython=True, cache=True)
def normalize_ip(vec: np.ndarray):
    vec /= np.linalg.norm(vec)


@jit(nopython=True, cache=True)
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
        dtype=dtype,
    )

@jit(nopython=True, cache=True)
def mat44_create_from_quaternion(quat: np.ndarray):
    ret = np.eye(4, 4)
    ret[0:3, 0:3] = mat33_create_from_quaternion(quat=quat)
    return ret

pyrr.Matrix44.inverse

@jit(nopython=True, cache=True)
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

@jit(nopython=True, cache=True)
def multiply_quaternions(q1:np.ndarray, q2:np.ndarray):

    q1x, q1y, q1z, q1w = q1
    q2x, q2y, q2z, q2w = q2

    # Not sure why the -1's need to be in here...
    # The pyrr implementation was used for this.
    return np.array(
        [
            q1x * q2w + q1y * q2z - q1z * q2y + q1w * q2x,
            -q1x * q2z + q1y * q2w + q1z * q2x + q1w * q2y,
            q1x * q2y - q1y * q2x + q1z * q2w + q1w * q2z,
            -q1x * q2x - q1y * q2y - q1z * q2z + q1w * q2w,
        ],
        dtype=q1.dtype
    )



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
    @jit(nopython=True, cache=True)
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
    from classes import Csys
    from scripts.classes import Csys as OldCsys
    @timer
    @jit(nopython=True, cache=True)
    def operate_on_csys1(csys:"Csys", x:np.ndarray, count=1):
        for i in range(count):
            x += csys.transformation_matrix
            csys.rzg(1.0)
            pass
        return x

    @timer
    def operate_on_csys2(csys:Csys, x:np.ndarray, count=1):
        for i in range(count):
            x += csys.transformation_matrix
            csys.rzg(1.0)
        return x

    x1 = Csys()
    x2 = OldCsys()

    print("Including Compilation:")
    x1a = np.zeros((4,4))
    x2a = np.zeros((4,4))
    operate_on_csys1(x1, x1a)
    operate_on_csys2(x2, x2a)

    print("")
    print("Excluding Compilation:")
    x1a = operate_on_csys1(x1, x1a, COUNT)
    x2a = operate_on_csys2(x2, x2a, COUNT // 100)

    pass


def update_mesh_pos_testing():
    import trimesh
    from classes import Triangle, Csys
    from scripts.classes import Triangle as OldTriangle
    from scripts.classes import Mesh, Material


    msh = trimesh.load("objects/tyre_tread.obj")

    myMesh = Mesh(material=Material())
    myMesh.csys.pos = pyrr.Vector3((10, 12, 0))
    myMesh.csys.rzg(30)
    # myMesh.csys.rxg(15)

    vertex_indices_arr = msh.faces.astype(np.int32)
    vertices = msh.vertices.astype(np.float32)
    vertex_normals = msh.vertex_normals.astype(np.float32)

    triangles_new = []
    triangles_old = []
    for i, vertex_indices in enumerate(vertex_indices_arr):
        posA, posB, posC = vertices[vertex_indices]
        normalA, normalB, normalC = vertex_normals[vertex_indices]
        triangles_new.append(
            Triangle(
                posA, posB, posC, normalA, normalB, normalC, 1, 1,
            )
        )
        triangles_old.append(
            OldTriangle(
                posA, posB, posC, normalA, normalB, normalC, myMesh
            )
        )

        if i % 100 == 0:
            print(f"  {i / len(vertex_indices_arr) * 100:.4f}%", end="\r")


    simulated_csys = Csys()
    simulated_csys._pos = np.array([10, 12, 0], dtype=np.float32)
    simulated_csys.rzg(30)
    # simulated_csys.rxg(15)

    @timer
    @jit(nopython=True, cache=True)
    def update_triangles_new(triangles:list[Triangle], csys:Csys):
        for i, triangle in enumerate(triangles):
            triangle.update_pos_to_csys(csys)
            # if i % 100 == 0:
            #     print("  {}%\r".format(i / len(vertex_indices_arr) * 100))

    @timer
    def update_triangles_old(triangles:list[OldTriangle]):
        for i, triangle in enumerate(triangles):
            triangle.update_pos_with_mesh2()
            if i % 100 == 0:
                print("  {}%".format(i / len(vertex_indices_arr) * 100), end="\r")



    print("Including Compilation:")
    update_triangles_new(triangles_new[:1], simulated_csys)
    update_triangles_old(triangles_old[:1])

    print("")
    print("Excluding Compilation:")
    update_triangles_new(triangles_new, simulated_csys)
    update_triangles_old(triangles_old)

    pass
