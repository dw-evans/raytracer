# from scripts.classes import (
#     Scene,
#     Mesh,
#     Triangle,
#     Sphere,
#     Material,
#     Csys,
# )

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

file = Path() / "objects/smooth_cube.obj"

mesh = trimesh.load(file)

data = []

for i, face in enumerate(mesh.faces):

    vertex_indices = face

    v0, v1, v2 = mesh.vertices[vertex_indices]
    n0, n1, n2 = mesh.vertex_normals[vertex_indices]

    pass
pass
