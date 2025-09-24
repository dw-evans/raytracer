from __future__ import annotations

from numba import jit, njit
from numba.types import float32, int32
from numba.experimental import jitclass

from . import functions
from .functions import (
    mat33_create_from_quaternion,
    mat44_create_from_quaternion,
    multiply_quaternions,
    normalize,
    normalize_ip,
    quaternion_create_from_axis_rotation
)

import numpy as np

spec_csys = [
    ("_pos", float32[:]),
    ("quat", float32[:]),
]

@jitclass(spec=spec_csys)
class Csys:
    """Numba-compiled Csys class"""
    
    def __init__(self) -> None:
        self._pos = np.array([0, 0, 0], dtype=np.float32)
        self.quat = np.array([0, 0, 0, 1], dtype=np.float32)

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, val):
        self._pos = val

    @property
    def rotation_matrix(self):
        return mat33_create_from_quaternion(self.quat)

    @property
    def transformation_matrix(self):
        ret = mat44_create_from_quaternion(self.quat)
        ret[3][0:3] = self._pos
        return ret
    
    def set_pos(self, p) -> Csys:
        val = np.array(p, dtype=np.float32)
        if not val.shape == (3,):
            raise ValueError("must provide (3,) vector")
        self._pos = val
        return self


    def rxg(self, degrees: float) -> Csys:
        rad = np.radians(degrees)
        ax = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.quat = normalize(multiply_quaternions(self.quat, quaternion_create_from_axis_rotation(ax, rad)))
        return self
    
    def ryg(self, degrees: float) -> Csys:
        rad = np.radians(degrees)
        ax = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.quat = normalize(multiply_quaternions(self.quat, quaternion_create_from_axis_rotation(ax, rad)))
        return self
    
    def rzg(self, degrees: float) -> Csys:
        rad = np.radians(degrees)
        ax = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.quat = normalize(multiply_quaternions(self.quat, quaternion_create_from_axis_rotation(ax, rad)))
        return self
    
    def rxp(self, degrees: float) -> Csys:
        rad = np.radians(degrees)
        ax = self.rotation_matrix[0][0:3]
        self.quat = normalize(multiply_quaternions(self.quat, quaternion_create_from_axis_rotation(ax, rad)))
        return self
    
    def ryp(self, degrees: float) -> Csys:
        rad = np.radians(degrees)
        ax = self.rotation_matrix[1][0:3]
        self.quat = normalize(multiply_quaternions(self.quat, quaternion_create_from_axis_rotation(ax, rad)))
        return self
    
    def rzp(self, degrees: float) -> Csys:
        rad = np.radians(degrees)
        ax = self.rotation_matrix[2][0:3]
        self.quat = normalize(multiply_quaternions(self.quat, quaternion_create_from_axis_rotation(ax, rad)))
        return self
    
    def tg(self, pos:np.ndarray) -> Csys:
        self._pos = pos.astype(np.float32)

    def txg(self, dst:float) -> Csys:
        self._pos[0] += dst
        return self
    
    def tyg(self, dst:float) -> Csys:
        self._pos[1] += dst
        return self
    
    def tzg(self, dst:float) -> Csys:
        self._pos[2] += dst
        return self
    
    def txp(self, dst:float) -> Csys:
        self._pos += dst * self.rotation_matrix[0][0:3]
        return self
    
    def typ(self, dst:float) -> Csys:
        self._pos += dst * self.rotation_matrix[1][0:3]
        return self
    
    def tzp(self, dst:float) -> Csys:
        self._pos += dst * self.rotation_matrix[2][0:3]
        return self
    


spec_triangle = [
    ("posA", float32[:]),
    ("posB", float32[:]),
    ("posC", float32[:]),
    ("posA0", float32[:]),
    ("posB0", float32[:]),
    ("posC0", float32[:]),
    ("normalA", float32[:]),  
    ("normalB", float32[:]),
    ("normalC", float32[:]),
    ("normalA0", float32[:]),  
    ("normalB0", float32[:]),
    ("normalC0", float32[:]),
    ("mesh_index", int32),
    ("id", int32),
]


ALL_TRIANGLES = []

# @staticmethod
# def get_all_triangles_arr():
#     return ALL_TRIANGLES

@jitclass(spec=spec_triangle)
class Triangle:
    # TRIANGLES = []
    def __init__(
        self,
        posA:np.ndarray,
        posB:np.ndarray,
        posC:np.ndarray,
        normalA:np.ndarray,
        normalB:np.ndarray,
        normalC:np.ndarray,
        mesh_parent_index:int,
        triangle_id:int,
    ) -> None:
        
        self.posA = posA
        self.posB = posB
        self.posC = posC

        self.posA0 = posA.copy()
        self.posB0 = posB.copy()
        self.posC0 = posC.copy()

        self.normalA = normalA
        self.normalB = normalB
        self.normalC = normalC

        self.normalA0 = normalA.copy()
        self.normalB0 = normalB.copy()
        self.normalC0 = normalC.copy()

        self.mesh_index = mesh_parent_index

        self.id = triangle_id



    def update_pos_to_csys(self, csys:Csys) -> Triangle:

        # Not sure why I need to intentionally type this...
        trans = csys.transformation_matrix.astype(np.float32)
        rot = trans[0:3, 0:3]

        self.posA = np.dot(np.array([*self.posA0, 1.0], dtype=np.float32), trans)[0:3]
        self.posB = np.dot(np.array([*self.posB0, 1.0], dtype=np.float32), trans)[0:3]
        self.posC = np.dot(np.array([*self.posC0, 1.0], dtype=np.float32), trans)[0:3]

        # self.posA, self.posB, self.posC = (
        #     np.dot(np.array([
        #         [*self.posA0, 1.0],
        #         [*self.posB0, 1.0],
        #         [*self.posC0, 1.0],
        #     ], dtype=np.float32),
        #     trans,
        #     )[:,:3]
        # )

        self.normalA = np.dot(self.normalA0, rot)
        self.normalB = np.dot(self.normalB0, rot)
        self.normalC = np.dot(self.normalC0, rot)

        # self.normalA, self.normalB, self.normalC = (
        #     np.dot(np.array([]))
        # )


        return self
    
    @property
    def positions(self):
        return self.posA, self.posB, self.posC
    
    @property
    def normals(self):
        return self.normalA, self.normalB, self.normalC
    
#     def tobytes(self):
#         bytes_data = (
#             self.posA.tobytes() +
#             self.posB.tobytes() + 
#             self.posC.tobytes() + 
#             self.normalA.tobytes() + 
#             self.normalB.tobytes() + 
#             self.normalC.tobytes() + 
#             self.mesh_index.tobytes() + 
#             self.triangle_id.tobytes()
#         )
#         return bytes_data
    
# @jit(nopython=True, cache=True)
# def many_triangles_to_bytes(triangles:list[Triangle]):
#     ret = b""
#     for i, triangle in enumerate(triangles):
#         ret += triangle.tobytes()


@jit(nopython=True, cache=True)
def update_triangles_to_csys(triangles:list[Triangle], csys:Csys):
    for i, triangle in enumerate(triangles):
        triangle.update_pos_to_csys(csys)

# @jit(nopython=False)
# def triangles_to_array(triangles:list[Triangle]) -> np.ndarray:
#     """Writes triangles to an array that can be sent straight to a buffer"""
#     pass
#     dtype = [
#         ("field00", "f4"),
#         ("field01", "f4"),
#         ("field02", "f4"),
#         ("field03", "f4"),
#         ("field04", "f4"),
#         ("field05", "f4"),
#         ("field06", "f4"),
#         ("field07", "f4"),
#         ("field08", "f4"),
#         ("field09", "f4"),
#         ("field10", "f4"),
#         ("field11", "f4"),
#         ("field12", "f4"),
#         ("field13", "f4"),
#         ("field14", "f4"),
#         ("field15", "f4"),
#         ("field16", "f4"),
#         ("field17", "f4"),
#         ("field18", "f4"),
#         ("field19", "f4"),
#         ("field20", "f4"),
#         ("field21", "f4"),
#         ("field22", "f4"),
#         ("field23", "i4"),
#         ("field24", "i4"),
#         ("field25", "f4"),
#         ("field26", "f4"),
#         ("field27", "f4"),
#     ]

#     # I think this line with the fancy type does not play nice
#     # with the jit
#     tri_data = np.zeros(
#         len(triangles),
#         dtype=dtype,
#     )

#     for i, tri in enumerate(triangles):
#         tri_data[i] = (
#             *tri.posA, 0.0,
#             *tri.posB, 0.0,
#             *tri.posC, 0.0,
#             *tri.normalA, 0.0,
#             *tri.normalB, 0.0,
#             *tri.normalC, tri.mesh_index,
#             tri.triangle_id, 0.0, 0.0, 0.0,
#         )

#     return tri_data


@jit(nopython=True, cache=True)
def triangles_to_array(triangles:list[Triangle]) -> np.ndarray:
    """Writes triangles to an array that can be sent straight to a buffer"""
    pass
    dtype = np.float32

    # I think this line with the fancy type does not play nice
    # with the jit
    tri_data = np.zeros(
        (len(triangles), 28),
        dtype=dtype,
    )

    pass
    for i, tri in enumerate(triangles):
        tri_data[i, :] = np.array([
            *tri.posA, 0.0,
            *tri.posB, 0.0,
            *tri.posC, 0.0,
            *tri.normalA, 0.0,
            *tri.normalB, 0.0,
            *tri.normalC, np.int32(tri.mesh_index).view(np.float32),
            np.int32(tri.id).view(np.float32), 0.0, 0.0, 0.0,
            ], 
            dtype=np.float32,
        )

    return tri_data


@jit(nopython=True, cache=True)
def create_and_register_triangles_from_obj_data(
    vertex_indices_arr: np.ndarray, vertices: np.ndarray, vertex_normals: np.ndarray, mesh_idx:int, triangle_id_start:int,
) -> list[Triangle]:
    ret = []
    for (i, vertex_indices) in enumerate(vertex_indices_arr):
        v0, v1, v2 = vertices[vertex_indices]
        n0, n1, n2 = vertex_normals[vertex_indices]
        ret.append(
            Triangle(
                v0,
                v1,
                v2,
                n0,
                n1,
                n2,
                mesh_idx,
                triangle_id_start + i,
            )
        )
    return ret


if __name__ == "__main__":
    x = Csys()
    x.rzg(10.0)
    x._pos = np.array([2, 5, 10])

        