from ..classes import (
    Scene,
    Mesh,
    Triangle,
    Sphere,
    Material,
    Csys,
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


material_red = Material(
    Vector4((1.0, 0.0, 0.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.1,
)
material_blue = Material(
    Vector4((0.0, 0.0, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.1,
)
material_green = Material(
    Vector4((0.0, 1.0, 0.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.1,
)
material_white = Material(
    Vector4((1.0, 0, 0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.1,
)

material_red_passthrough = Material(
    Vector4((1.0, 0.0, 0.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.1,
    transparent_from_behind=True
)
material_blue_passthrough = Material(
    Vector4((0.0, 0.0, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.1,
    transparent_from_behind=True
)
material_green_passthrough = Material(
    Vector4((0.0, 1.0, 0.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.1,
    transparent_from_behind=True
)
material_white_passthrough = Material(
    Vector4((1.0, 1.0, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.1,
    transparent_from_behind=True
)

material_light_source_1 = Material(
    Vector4((0.0, 0.0, 0.0, 1.0), dtype="f4"),
    emissionColor=Vector3((1, 1, 1), dtype="f4"),
    emissionStrength=4.0,
)

glass_material = Material(
    color=Vector4((1.0, 0.9, 0.9, 1.0)),
    smoothness=1.0,
    transmission=0.9,
    ior=1.6,
)

atmosphere_material = Material(
    Vector4((1.0,1.0, 1.0, 1.0),  dtype="f4"),
    transmission=1.0,
    ior=1.0,
)

scene = Scene()
scene.atmosphere_material = atmosphere_material

meshes = []


material_subject = Material(
    color=Vector4((1.0, 0.3, 0.6, 1.0)),
    smoothness=1.0,
    transmission=0.9,
    ior=1.6,
)

load_data = [
    ("objects/old/final_scene/wall_left.obj", material_red_passthrough),
    ("objects/old/final_scene/wall_right.obj", material_green_passthrough),
    # ("objects/old/final_scene/wall_top.obj", material_white_passthrough),
    ("objects/old/final_scene/wall_bottom.obj", material_white_passthrough),
    ("objects/old/final_scene/wall_front.obj", material_blue_passthrough),
    ("objects/old/final_scene/wall_back.obj", material_white_passthrough),
    ("objects/old/final_scene/light.obj", material_light_source_1),
    # ("objects/old/final_scene/subject.obj", material_subject),
    ("objects/old/heart.obj", material_subject),

]

car_csys = numba_scripts.classes.Csys()
car_csys._pos = np.array([0.0, 1.0, 8.0], dtype=np.float32)
car_csys.ryg(180)

scene.cam.csys.pos = pyrr.Vector3([0.0, 2.0, 0.0])

for (i, (f, material)) in enumerate(load_data):
    msh = trimesh.load(f)

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

    msh1 = Mesh(material=material)

    msh1.csys = car_csys

    msh1.triangles = triangles

    scene.meshes.append(msh1)

print(f"There are `{scene.count_triangles()}` triangles in the scene.")
