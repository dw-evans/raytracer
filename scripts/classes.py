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
        self.origin: Vector3 = Vector3((0.0, 0.0, 0.0))
        self.dir = Vector3((0.0, 0.0, 0.0))


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
                "3f4x 3f4x 3f4x 3f4x 3f4x 3f i",
                *self.posA.astype("f4"),
                *self.posB.astype("f4"),
                *self.posC.astype("f4"),
                *self.normalA.astype("f4"),
                *self.normalB.astype("f4"),
                *self.normalC.astype("f4"),
                self.parent.mesh_index,
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

        self.posA = Vector3(
            pyrr.matrix44.apply_to_vector(
                self.parent.csys.transformation_matrix, self.posA0
            )
        )
        self.posB = Vector3(
            pyrr.matrix44.apply_to_vector(
                self.parent.csys.transformation_matrix, self.posB0
            )
        )
        self.posC = Vector3(
            pyrr.matrix44.apply_to_vector(
                self.parent.csys.transformation_matrix, self.posC0
            )
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
    # an arbitrary counter that keeps a UID for each mesh
    MESH_INDEX = 0

    def __init__(self) -> None:
        self.pos: Vector3 = Vector3((0.0, 0.0, 0.0))
        self.triangles: list[Triangle] = []
        self.csys = Csys()
        self.mesh_index = self.MESH_INDEX

        Mesh.MESH_INDEX += 1

    def tobytes(self) -> bytes | bytearray:

        bbox = self.bounding_box
        return struct.pack(
            "i12x 3f4x 3f4x",
            self.mesh_index,
            *bbox[0],
            *bbox[1],
        )

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
                    posB=v1,
                    posC=v2,
                    normalA=normal,
                    normalB=normal,
                    normalC=normal,
                    parent=None,
                    material=material,
                )
            )
        msh.add_tri(tris)
        return msh

    @property
    def bounding_box(self) -> pyrr.aabb:
        pts = []
        for tri in self.triangles:
            pts += tri.positions
        ret = [Vector3(x) for x in pyrr.aabb.create_from_points(pts)]

        return ret


class Camera:
    def __init__(self) -> None:
        self.fov = 30.0  # degrees
        self.aspect = 16.0 / 9.0
        self.near_plane = 240.0
        # self.pos = Vector3((0.0, 0.0, 0.0), dtype="f4")

        # TODO investigate these, they don't match intent
        # self.roll = 0.0
        # self.pitch = 0.0
        # self.yaw = 0.0

        self.csys = Csys()

    @property
    def orientation(self):
        """Returns the mat44 based on the pitch, roll, yaw angles.
        TODO: support the non-global implementation of this, whatever its called
        i.e. sequential rotations like this: RzRxRz"""  #

        raise NotImplementedError
        # return self.csys.rotation_matrix

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
        return self.csys.transformation_matrix.inverse
        # return self.orientation.inverse * Matrix44.from_translation(-1 * self.pos)
        Matrix44.identity() * Matrix44.from_translation(-1 * self.pos)

    @property
    def view_params(self):
        return Vector3(
            [self.plane_width, self.plane_height, self.near_plane],
        )


class Csys:
    def __init__(self):
        self.pos = Vector3((0.0, 0.0, 0.0))
        self.quat = pyrr.Quaternion()

    @property
    def rotation_matrix(self) -> Matrix33:
        """returns just the orientation matrix"""
        # raise NotImplementedError
        return Matrix33(pyrr.matrix33.create_from_quaternion(self.quat))

    @property
    def transformation_matrix(self) -> Matrix44:
        """Returns the rotation translation Mat44 from the postion and quat"""
        ret = pyrr.matrix44.create_from_quaternion(self.quat)
        ret[3][0:3] = self.pos
        ret = Matrix44(ret, dtype="f4")
        return ret

    def __str__(self) -> str:
        return self.transformation_matrix.__str__()

    def __repr__(self) -> str:
        scale, rotate, translate = self.transformation_matrix.decompose()
        return f"Csys(\nscale={scale}\nrotate={rotate}\ntranslate={translate}\n{self.transformation_matrix.__str__()}\n)"

    @property
    def scale(self):
        # TODO implement scaling to the meshes in this way potentially?
        raise NotImplementedError
        scale, _ = self.transformation_matrix.decompose()
        return scale

    def rxp(self, degrees: float) -> Csys:
        """Rotate about x' axis"""
        rad = radians(degrees)
        ax = Vector3(self.rotation_matrix.r1)
        self.quat = self.quat.cross(pyrr.quaternion.create_from_axis_rotation(ax, rad))
        return self

    def ryp(self, degrees: float) -> Csys:
        """Rotate about y' axis"""
        rad = radians(degrees)
        ax = Vector3(self.rotation_matrix.r2)
        self.quat = self.quat.cross(pyrr.quaternion.create_from_axis_rotation(ax, rad))
        return self

    def rzp(self, degrees: float) -> Csys:
        """Rotate about z' axis"""
        rad = radians(degrees)
        ax = Vector3(self.rotation_matrix.r3)
        self.quat = self.quat.cross(pyrr.quaternion.create_from_axis_rotation(ax, rad))
        return self

    def rxg(self, degrees: float) -> Csys:
        """Rotate about x global axis"""
        rad = radians(degrees)
        ax = Vector3((1.0, 0.0, 0.0))
        self.quat = self.quat.cross(pyrr.quaternion.create_from_axis_rotation(ax, rad))
        return self

    def ryg(self, degrees: float) -> Csys:
        """Rotate about y global axis"""
        rad = radians(degrees)
        ax = Vector3((0.0, 1.0, 0.0))
        self.quat = self.quat.cross(pyrr.quaternion.create_from_axis_rotation(ax, rad))
        return self

    def rzg(self, degrees: float) -> Csys:
        """Rotate about z global axis"""
        rad = radians(degrees)
        ax = Vector3((0.0, 0.0, 1.0))
        self.quat = self.quat.cross(
            Vector4(pyrr.quaternion.create_from_axis_rotation(ax, rad))
        )
        return self

    def tg(self, vec: Vector3) -> Csys:
        """Translate incrementally global"""
        x = vec
        ax = Matrix33.identity()
        self.pos = ax * x
        return self

    def tp(self, vec: Vector3) -> Csys:
        """Translate incrementally local"""
        x = vec
        ax = self.rotation_matrix
        self.pos = ax * x
        return self

    def txp(self, dst: float) -> Csys:
        ax = self.rotation_matrix.r1[:3]
        self.pos += dst * ax
        return self

    def typ(self, dst: float) -> Csys:
        ax = self.rotation_matrix.r2.xyz
        self.pos += dst * ax
        return self

    def tzp(self, dst: float) -> Csys:
        ax = self.rotation_matrix.r3.xyz
        self.pos += dst * ax
        return self

    def txg(self, dst: float) -> Csys:
        self.pos.x += dst
        return self

    def tyg(self, dst: float) -> Csys:
        self.pos.y += dst
        return self

    def tzg(self, dst: float) -> Csys:
        self.pos.z += dst
        return self

    # def move_to(self, vec: Vector3) -> Csys:
    #     self.pos = vec
    #     return self

    # def txp(self, dst: float) -> Csys:
    #     """translate"""
    #     ax = self.rotation_matrix.r1
    #     self.pos += Vector3(ax[:3]).normalized * dst
    #     return self


class Scene:
    def __init__(self) -> None:
        self.meshes: list[Mesh] = []
        self.spheres: list[Sphere] = []
        self.cam = Camera()
        self.selected_objects: list[Triangle | Mesh | Sphere] = []

    def get_mesh_index(self, msh: Mesh) -> int:
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
    def triangles(self) -> list[Triangle]:
        ret = []
        for m in self.meshes:
            ret += m.triangles
        return ret

    def select_object(self, obj):
        self.selected_objects.append(obj)

    def deselect_object(self, obj):
        self.selected_objects.remove(obj)

    def deselect_all_objects(self):
        raise NotImplementedError
        self.selected_objects = []

    def serialise(self):
        raise NotImplementedError
        pass

    def serialiser(self) -> str:
        raise NotImplementedError
        pass
