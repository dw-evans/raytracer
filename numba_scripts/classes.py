from __future__ import annotations

from numba import jit
from numba.types import float32, int32
from numba.experimental import jitclass

import functions
from functions import (
    mat33_create_from_quaternion,
    mat44_create_from_quaternion,
    multiply_quaternions,
    normalize,
    normalize_ip,
    quaternion_create_from_axis_rotation
)

import numpy as np

spec_csys = [
    ("pos", float32[:]),
    ("quat", float32[:]),
]

@jitclass(spec=spec_csys)
class Csys:
    """Numba-compiled Csys class"""
    
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
    
    def txg(self, dst:float) -> Csys:
        self.pos[0] += dst
        return self
    
    def tyg(self, dst:float) -> Csys:
        self.pos[1] += dst
        return self
    
    def tzg(self, dst:float) -> Csys:
        self.pos[2] += dst
        return self
    
    def txp(self, dst:float) -> Csys:
        self.pos += dst * self.rotation_matrix[0][0:3]
        return self
    
    def typ(self, dst:float) -> Csys:
        self.pos += dst * self.rotation_matrix[1][0:3]
        return self
    
    def tzp(self, dst:float) -> Csys:
        self.pos += dst * self.rotation_matrix[2][0:3]
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
    ("triangle_id", int32),
]


@jitclass(spec=spec_triangle)
class Triangle:
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

        self.triangle_id = triangle_id
    

    def update_pos_to_csys(self, csys:Csys) -> Triangle:

        # Not sure why I need to intentionally type this...
        trans = csys.transformation_matrix.astype(np.float32)
        rot = trans[0:3, 0:3]

        self.posA = np.dot(np.array([*self.posA0, 1.0]).astype(np.float32), trans)[0:3]
        self.posB = np.dot(np.array([*self.posB0, 1.0]).astype(np.float32), trans)[0:3]
        self.posC = np.dot(np.array([*self.posC0, 1.0]).astype(np.float32), trans)[0:3]

        self.normalA = np.dot(self.normalA0, rot)
        self.normalB = np.dot(self.normalB0, rot)
        self.normalC = np.dot(self.normalC0, rot)

        return self



if __name__ == "__main__":
    x = Csys()
    x.rzg(10.0)
    x.pos = np.array([2, 5, 10])

        