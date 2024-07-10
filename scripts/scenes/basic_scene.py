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

scene = Scene()

material_plain_1 = Material(
    Vector4((1.0, 0.5, 0.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    0.0,
    smoothness=0.0,
)
material_plain_2 = Material(
    Vector4((1.0, 1.0, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    0.0,
    smoothness=0.6,
)
material_plain_3 = Material(
    Vector4((0.5, 0.0, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    0.0,
    smoothness=0.0,
)
material_plain_4 = Material(
    Vector4((1.0, 0, 0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    0.3,
    smoothness=0.2,
)
material_plain_5 = Material(
    Vector4((0.0, 1.0, 0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    0.3,
    smoothness=0.2,
)
material_light_source_1 = Material(
    Vector4((0.0, 0.0, 0.0, 1.0), dtype="f4"),
    Vector3((1, 1, 1), dtype="f4"),
    5.0,
)
material_light_source_2 = Material(
    Vector4((0.0, 0.0, 0.0, 1.0), dtype="f4"),
    Vector3((1, 1, 1), dtype="f4"),
    5.0,
)
highlight_material = Material(
    Vector4((0.3, 0.6, 0.8, 1.0), dtype="f4"),
    Vector3((0, 0, 0), dtype="f4"),
    0.0,
    smoothness=0.5,
)

spheres = [
    Sphere(
        pos=Vector3((0.0, -1001.0, 8), dtype="f4"),
        radius=1000.0,
        material=material_plain_3,
    ),
    Sphere(
        pos=Vector3((0.0, 3.0, 22), dtype="f4"),
        radius=10.0,
        material=material_plain_1,
    ),
    Sphere(
        pos=Vector3((2, 3, 4), dtype="f4"),
        radius=1,
        material=material_light_source_1,
    ),
    Sphere(
        pos=Vector3((-2, 3, 4), dtype="f4"),
        radius=1,
        material=material_light_source_1,
    ),
    Sphere(
        pos=Vector3((1, 0.5, 0), dtype="f4"),
        radius=0.5,
        material=material_light_source_2,
    ),
    Sphere(
        pos=Vector3((0, 0, 0), dtype="f4"),
        radius=0.2,
        material=material_plain_1,
    ),
]

scene.spheres = spheres

stl_file = Path() / "objects/funky_cube.stl"

msh0 = Mesh.from_stl(
    stl_file,
    material=material_plain_4,
)
scene.meshes.append(msh0)
msh0.csys.tp(Vector3((0, 0.0, 6.0)))
msh0.csys.rzg(180)

msh1 = Mesh.from_stl(
    stl_file,
    material=material_plain_4,
)
scene.meshes.append(msh1)

msh1.csys.tp(Vector3((-1.5, 1.0, 9.0)))
msh1.csys.rzg(100)

msh2 = Mesh.from_stl(
    stl_file,
    material=material_plain_4,
)
scene.meshes.append(msh2)
msh2.csys.tp(Vector3((2, 0.0, 8.0)))
msh2.csys.rzg(90)


# scene2 = Scene()
# material = Material(
#     Vector4((1.0, 0.0, 0.0, 1.0), dtype="f4"),
#     Vector3((0.0, 0.0, 0.0), dtype="f4"),
#     0.0,
#     smoothness=0.0,
# )

# tris = [
#     Triangle(
#         Vector3((-1, 0, 6)),
#         Vector3((0, 0, 6)),
#         Vector3((0, 1, 6)),
#         material=material,
#     ),
#     Triangle(
#         Vector3((0, 0, 6)),
#         Vector3((1, 0, 6)),
#         Vector3((0, 1, 6)),
#         material=material,
#     ),
# ]

# scene2.spheres = spheres

# msh = Mesh()

# msh.add_tri(tris)
# scene2.meshes.append(msh)

# pass


scene3 = Scene()

cube_mesh = Mesh.from_obj(
    Path() / "objects/smooth_cube.obj",
    material=material_plain_3,
)

cube_mesh.csys.tzg(6)
cube_mesh.csys.ryg(250)

scene3.meshes.append(cube_mesh)

scene3.spheres = spheres
