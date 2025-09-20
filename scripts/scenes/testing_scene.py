from ..classes import (
    Scene,
    Mesh,
    Triangle,
    Sphere,
    Material,
    # Csys,
)

import pyrr
from pyrr import (
    Vector3,
    Vector4,
    Matrix33,
    Matrix44,
    Quaternion,
)

from pathlib import Path

import trimesh
import numpy as np
import numba_scripts.classes
from numba_scripts.classes import Csys


material_plain_1 = Material(
    Vector4((1.0, 0.5, 0.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    0.0,
    smoothness=0.0,
)

MATERIALS = {
    "red": Material(
        color=Vector4((0.8, 0, 0, 1), dtype="f4"),
        emissionColor=Vector3((0.0, 0.0, 0.0), dtype="f4"),
        emissionStrength=0.0,
        smoothness=0.0,
        transparent_from_behind=False,
    ),
    "green": Material(
        color=Vector4((0.0, 0.8, 0, 1), dtype="f4"),
        emissionColor=Vector3((0.0, 0.0, 0.0), dtype="f4"),
        emissionStrength=0.0,
        smoothness=0.0,
        transparent_from_behind=False,
    ),
    "blue": Material(
        color=Vector4((0.0, 0.0, 0.8, 1), dtype="f4"),
        emissionColor=Vector3((0.0, 0.0, 0.0), dtype="f4"),
        emissionStrength=0.0,
        smoothness=0.0,
        transparent_from_behind=False,
    ),
    "white_100": Material(
        color=Vector4((1, 1, 1, 1), dtype="f4"),
        emissionColor=Vector3((0.0, 0.0, 0.0), dtype="f4"),
        emissionStrength=0.0,
        smoothness=0.0,
        transparent_from_behind=False,
    ),
    "mirror-one-sided": Material(
        color=Vector4((1, 1, 1, 1), dtype="f4"),
        emissionColor=Vector3((0.0, 0.0, 0.0), dtype="f4"),
        emissionStrength=0.0,
        smoothness=1.0,
        transparent_from_behind=True,
    ),
    "white-light-1-transparent": Material(
        color=Vector4((0, 0, 0, 1), dtype="f4"),
        emissionColor=Vector3((1.0, 1.0, 1.0), dtype="f4"),
        emissionStrength=1.0,
        smoothness=1.0,
        transparent_from_behind=True,
    ),
    "white-light-2": Material(
        color=Vector4((0, 0, 0, 1), dtype="f4"),
        emissionColor=Vector3((1.0, 1.0, 1.0), dtype="f4"),
        emissionStrength=1.0,
        smoothness=1.0,
        transparent_from_behind=False,
        
    ),
    "atmosphere": Material(
        color=Vector4((0.2, 0.2, 0.2, 1), dtype="f4"),
        emissionColor=Vector3((1.0, 1.0, 1.0), dtype="f4"),
        emissionStrength=1.0,
        smoothness=1.0,
        transmission=1.0,
        ior=1.0
    )
}

spheres = [
    Sphere(
        pos=Vector3((0.0, -1001.0, 8), dtype="f4"),
        radius=1000.0,
        material=MATERIALS["blue"],
    ),
    Sphere(
        pos=Vector3((0.0, 3.0, 22), dtype="f4"),
        radius=10.0,
        material=MATERIALS["red"],
    ),
    Sphere(
        pos=Vector3((5, 5, 0), dtype="f4"),
        radius=3,
        material=material_light_source_1,
    ),
    Sphere(
        pos=Vector3((0, 0, 0), dtype="f4"),
        radius=0.2,
        material=material_plain_1,
    ),
]


scene = Scene()

scene.atmosphere_material = MATERIALS["atmosphere"]

DEFAULT_CSYS = Csys()

scene.cam.csys.pos = pyrr.Vector3([0, 1.0, 0], dtype="f2")

car_csys = numba_scripts.classes.Csys()
car_csys.set_pos([0.0, 0.0, 8.0])
car_csys.ryg(180-45)


fname_material_pose = [
    # ("objects/test.blend/cube.obj", hubcap_material, car_csys),
]


for i, (_fname, _material, _csys) in enumerate(fname_material_pose):
    msh = trimesh.load(_fname)

    vertex_indices_arr = msh.faces.astype(np.int32)
    vertices = msh.vertices.astype(np.float32)
    vertex_normals = msh.vertex_normals.astype(np.float32)

    triangle_count_start = scene.count_triangles()
    mesh_idx = i
    triangles = numba_scripts.classes.triangles_from_obj_data(
        vertex_indices_arr,
        vertices,
        vertex_normals,
        mesh_idx,
        triangle_count_start,
    )

    msh1 = Mesh(material=_material)
    # msh1.csys = car_csys
    msh1.csys = _csys
    msh1.triangles = triangles
    scene.meshes.append(msh1)

print(f"There are `{scene.count_triangles()}` triangles in the scene.")



from ..animate import Animation
from math import sin, cos, tan, pi

import numba_scripts.classes
def mesh_csys_animate(obj: numba_scripts.classes.Csys, t):
    obj.pos[0] = 0.0 + 10.0 * sin(t / 2)
    obj.pos[1] = -0.5
    obj.pos[2] = 6.0 + 5.0 * sin(t / 3)

# scene.animations.append(Animation(scene.meshes[0].csys))


# def camera_csys_animate(obj: Csys, t):
#     # view_mat = Matrix44(
#     #     pyrr.matrix44.create_look_at(obj.pos, msh1.csys.pos, Vector3((0.0, 1.0, 0.0)))
#     # )

#     # eye = Vector3((0.0, 0.0, 0.0))
#     # target = obj.pos - msh1.csys.pos
#     # up = Vector3((0.0, 1.0, 0.0))

#     # view_mat = Matrix44(pyrr.matrix44.create_look_at(eye, target, up))

#     z_axis = (msh1.csys.pos - obj.pos).normalised
#     x_axis = Vector3((0.0, 1.0, 0.0)).cross(z_axis)
#     y_axis = z_axis.cross(x_axis)

#     obj.quat.xyzw = pyrr.quaternion.create_from_matrix(
#         Matrix33((x_axis, y_axis, z_axis))
#     )


# scene.animations.append(Animation(msh1.csys, mesh_csys_animate))
# scene.animations.append(Animation(scene.cam.csys, camera_csys_animate))
