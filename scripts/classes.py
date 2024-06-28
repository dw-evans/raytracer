from __future__ import annotations

from typing import Protocol, Iterable
from pyrr import Vector4, Vector3, Matrix44, Matrix33
import struct
import numpy as np
from numpy import radians, tan, cos, sin
from stl import mesh
import copy
import pyrr


class HitInfo:
    def __init__(self) -> None:
        self.didHit: bool = False
        self.dst: float = np.inf
        self.hitPoint = np.array([0.0, 0.0, 0.0])
        self.normal = np.array([0.0, 0.0, 0.0])


class Ray:
    def __init__(self) -> None:
        self.origin = np.array([0.0, 0.0, 0.0])
        self.dir = np.array([0.0, 0.0, 0.0])


class ByteableObject(Protocol):
    def tobytes(self) -> bytes | bytearray: ...


class Material(ByteableObject):
    def __init__(
        self,
        color: Vector4 = Vector4((0.0, 0.0, 0.0, 0.0)),
        emissionColor: Vector3 = Vector3((0.0, 0.0, 0.0)),
        emissionStrength: float = 0.0,
        smoothness: float = 0.0,
    ) -> None:

        self.color = color
        self.emissionColor = emissionColor
        self.emissionStrength = emissionStrength
        self.smoothness = smoothness

    def tobytes(self):
        return struct.pack(
            "4f 3f f f12x",
            *self.color,
            *self.emissionColor,
            self.emissionStrength,
            self.smoothness,
        )


class Sphere(ByteableObject):
    def __init__(self, pos: Vector3, radius: float, material: Material) -> None:
        self.pos = pos
        self.radius = radius
        self.material = material

    def tobytes(self):
        return struct.pack("3f f", *self.pos, self.radius) + self.material.tobytes()


class Triangle(ByteableObject):
    def __init__(
        self,
        posA: Vector3,
        posB: Vector3,
        posC: Vector3,
        material: Material,
        normalA: Vector3 = None,
        normalB: Vector3 = None,
        normalC: Vector3 = None,
        parent: Mesh = None,
    ) -> None:

        self.posA = posA
        self.posB = posB
        self.posC = posC

        self.posA0 = copy.copy(posA)
        self.posB0 = copy.copy(posB)
        self.posC0 = copy.copy(posC)

        self.material = material

        self.normalA: Vector3 = normalA
        self.normalB: Vector3 = normalB
        self.normalC: Vector3 = normalC

        self.normalA0 = copy.copy(normalA)
        self.normalB0 = copy.copy(normalB)
        self.normalC0 = copy.copy(normalC)

        self.parent = parent

        self.__post_init__()

        pass

    def __post_init__(self) -> None:
        # Correct the normals based on the values provided
        self.normalA = (
            self.normal_from_vertices if self.normalA is None else self.normalA
        )
        self.normalB = (
            self.normal_from_vertices if self.normalB is None else self.normalB
        )
        self.normalC = (
            self.normal_from_vertices if self.normalC is None else self.normalC
        )

    @property
    def positions(self):
        return [self.posA, self.posB, self.posC]

    @property
    def normals(self):
        return [self.normalA, self.normalB, self.normalC]

    @property
    def positions0(self):
        return [self.posA0, self.posB0, self.posC0]

    @property
    def normals0(self):
        return [self.normalA0, self.normalB0, self.normalC0]

    @property
    def normal_from_vertices(self) -> Vector3:
        edge_ab = self.posB - self.posA
        edge_ac = self.posC - self.posA
        ret = edge_ab.cross(edge_ac)
        ret.normalize()
        return ret

    def tobytes(self) -> bytes | bytearray:
        return (
            struct.pack(
                "3f4x 3f4x 3f4x 3f4x 3f4x 3f4x",
                *self.posA.astype("f4"),
                *self.posB.astype("f4"),
                *self.posC.astype("f4"),
                *self.normalA.astype("f4"),
                *self.normalB.astype("f4"),
                *self.normalC.astype("f4"),
            )
            + self.material.tobytes()
        )

    def update_pos_with_mesh(self) -> Triangle:
        """Moves the position of the triangle to match that of the mesh"""
        if self.parent is None:
            print("Nothing to update")

        vec = self.parent.pos
        self.posA = self.posA0 + vec
        self.posB = self.posB0 + vec
        self.posC = self.posC0 + vec

        return self

    def update_pos_with_mesh2(self) -> Triangle:
        """Moves the position of the triangle to match that of the mesh"""
        if self.parent is None:
            print("Nothing to update")
            return self

        # vec = self.parent.csys.pos
        # self.posA = self.posA0 + vec
        # self.posB = self.posB0 + vec
        # self.posC = self.posC0 + vec

        self.posA = pyrr.matrix44.apply_to_vector(
            self.parent.csys.transformation_matrix, self.posA
        )
        self.posB = pyrr.matrix44.apply_to_vector(
            self.parent.csys.transformation_matrix, self.posB
        )
        self.posC = pyrr.matrix44.apply_to_vector(
            self.parent.csys.transformation_matrix, self.posC
        )

        return self

    def translate(self, vec: Vector3) -> Triangle:
        self.posA = self.posA0 + vec
        self.posB = self.posB0 + vec
        self.posC = self.posC0 + vec
        return self


# class AxisAlignedBox(ByteableObject):
#     def __init__(self, v0: Vector3, v1: Vector3) -> None:
#         self.v0 = v0
#         self.v1 = v1


class Mesh(ByteableObject):

    def __init__(self) -> None:
        self.pos: Vector3 = Vector3((0.0, 0.0, 0.0))
        self.triangles: list[Triangle] = []

        # self.orientation = Matrix44.identity()

        self.csys = Csys()

    def tobytes(self) -> bytes | bytearray:

        tri_bytes = b""
        for tri in self.triangles:
            tri_bytes += tri.tobytes()

        return struct.pack("3f4x", self.pos.astype("f4")) + tri_bytes

    # @property
    # def bounding_box(self) -> any:
    #     """Calculates the bounding box representation of the mesh"""
    #     pass

    def add_tri(self, tris: Triangle | Iterable[Triangle]):
        """Method to add triangles and maintain parent mesh relationship"""
        if isinstance(tris, Iterable):
            for tri in tris:
                tri.parent = self
                self.triangles.append(tri)
        else:
            tris.parent = self
            self.triangles.append(tris)
        return self

    # The triangles still don't seem to be working correctly.
    @classmethod
    def from_stl(self, file: str, material: Material = Material()):
        """Creates a mesh from a stl file"""
        mesh_data = mesh.Mesh.from_file(file)
        msh = Mesh()
        tris = []
        for facet in mesh_data.data:
            facet: np.ndarray
            normal = Vector3(facet[0])
            normal.normalize()
            v0, v1, v2 = (Vector3(x) for x in facet[1])
            tris.append(
                Triangle(
                    posA=v0,
                    posB=v2,
                    posC=v1,
                    normalA=normal,
                    normalB=normal,
                    normalC=normal,
                    parent=None,
                    material=material,
                )
            )
        msh.add_tri(tris)
        return msh

    def create_transformed_triangle(
        self, tris: Triangle | list[Triangle]
    ) -> Triangle | list[Triangle]:
        raise Exception
        if isinstance(tris, Iterable):
            ret = []
            for tri in tris:
                t_copy = copy.deepcopy(tri)
                for x in t_copy.positions:
                    x += self.pos
                ret.append(t_copy)
        else:
            tri = tris
            t_copy = copy.deepcopy(tri)
            for x in t_copy.positions:
                x += self.pos

        return ret

    def to_bytes_with_transforms(self) -> bytes:
        """Lazily writes the triangles in the mesh to bytes including the position of the
        mesh centroid
        TODO implement rotations etc to flex"""
        raise NotImplementedError
        ret = b""
        # for every triangle, deepcopy then write to bytes
        for t in self.triangles:
            t_copy = copy.deepcopy(t)
            for x in t_copy.positions:
                x += self.pos
            ret += t.tobytes()

        return ret


class Camera:
    def __init__(self) -> None:
        self.fov = 30.0  # degrees
        self.aspect = 16.0 / 9.0
        self.near_plane = 240.0
        self.pos = Vector3((0.0, 0.0, 0.0), dtype="f4")

        # TODO investigate these, they don't match intent
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    @property
    def orientation(self):
        """Returns the mat44 based on the pitch, roll, yaw angles.
        TODO: support the non-global implementation of this, whatever its called
        i.e. sequential rotations like this: RzRxRz"""
        return Matrix44.from_eulers(
            [radians(x) for x in [self.pitch, self.yaw, self.roll]]
        )

    @property
    def plane_height(self):
        """Returns the plane height based on the fov and near plane"""
        return self.near_plane * tan(radians(self.fov) * 0.5) * 2.0

    @property
    def plane_width(self):
        """Returns the width of the plane based on its height and aspect"""
        return self.plane_height * self.aspect

    @property
    def local_to_world_matrix(self):
        """Calculates the local to world mat44 transform"""
        return self.orientation.inverse * Matrix44.from_translation(-1 * self.pos)

    @property
    def view_params(self):
        return Vector3(
            [self.plane_width, self.plane_height, self.near_plane],
        )

    def rotate(self, axis: str, rot_deg: int | float | Iterable[int | float]):
        raise NotImplementedError
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


class Csys:
    def __init__(self):
        self.mat = Matrix44.identity()
        self.pos = Vector3((0.0, 0.0, 0.0))
        self.quat = pyrr.quaternion.create()

    @property
    def rotation_matrix(self):
        return Matrix33(pyrr.matrix33.create_from_quaternion(self.quat))

    @property
    def transformation_matrix(self):
        ret = pyrr.matrix44.create_from_quaternion(self.quat)
        ret[3][0:3] = self.pos
        ret = Matrix44(ret)
        return ret

    def __str__(self):
        return self.mat.__str__()

    def __repr__(self):
        scale, rotate, translate = self.mat.decompose()
        return f"Csys(\nscale={scale}\nrotate={rotate}\ntranslate={translate}\n{self.mat.__str__()}\n)"

    def Rx(self, degrees: float):
        rad = radians(degrees)
        ax = Vector3(self.rotation_matrix.r1)
        self.quat = pyrr.quaternion.create_from_axis_rotation(ax, rad)
        rot_mat = pyrr.matrix44.create_from_quaternion(self.quat)
        self.mat = pyrr.matrix44.multiply(self.mat, rot_mat)
        return self

    def Ry(self, degrees: float):
        rad = radians(degrees)
        ax = Vector3(self.rotation_matrix.r2)
        self.quat = pyrr.quaternion.create_from_axis_rotation(ax, rad)
        rot_mat = pyrr.matrix44.create_from_quaternion(self.quat)
        self.mat = pyrr.matrix44.multiply(self.mat, rot_mat)
        return self

    def Rz(self, degrees: float):
        rad = radians(degrees)
        ax = Vector3(self.rotation_matrix.r3)
        self.quat = pyrr.quaternion.create_from_axis_rotation(ax, rad)
        rot_mat = pyrr.matrix44.create_from_quaternion(self.quat)
        self.mat = pyrr.matrix44.multiply(self.mat, rot_mat)
        return self

    def translate(self, vec: Vector3):
        x, y, z = vec
        self.pos.x += x
        self.pos.y += y
        self.pos.z += z
        return self

    # def convert_point(self, pt:Vector3):


class Scene:
    def __init__(self) -> None:
        self.meshes: list[Mesh] = []
        self.spheres: list[Sphere] = []
        self.cam = Camera()

    def get_mesh_index(self, msh: Mesh) -> int:
        raise Exception
        try:
            return self.meshes.index(msh)
        except:
            raise Exception("Mesh not in list")

    def count_triangles(self) -> int:
        count = 0
        for mesh in self.meshes:
            count += len(mesh.triangles)
        return count

    @property
    def triangles(self) -> list[triangles]:
        ret = []
        for m in self.meshes:
            ret += m.triangles
        return ret
