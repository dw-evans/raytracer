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
    # Sphere(
    #     pos=Vector3((3, 3, 1), dtype="f4"),
    #     radius=0.5,
    #     material=material_plain_4,
    # ),
    # Sphere(
    #     pos=Vector3((-2, 3, 4), dtype="f4"),
    #     radius=1,
    #     material=material_plain_4,
    # ),
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


scene.spheres = spheres

glass_material = Material(
    color=Vector4((1.0, 0.9, 0.9, 1.0)),
    smoothness=1.0,
    transmission=0.9,
    ior=1.6,
)


# msh1 = Mesh.from_stl(
#     Path() / "objects/gem.stl",
#     material=glass_material,
# )
# msh1.csys.tp(Vector3((0, -0.5, 6.0)))
# msh1.csys.rzg(270)
# msh1.csys.rxg(90)

msh1 = Mesh.from_obj(
    Path() / "objects/smooth_disc.obj",
    material=glass_material,
)
msh1.csys.tp(Vector3((0, -0.5, 6.0)))
# msh1.csys.rzg(-90)
msh1.csys.ryg(90)
# msh1.csys.rxg(90)

scene.meshes.append(msh1)


atmosphere_material = Material(
    Vector4(
        (
            1.0,
            1.0,
            1.0,
            1.0,
        ),
        dtype="f4",
    ),
    transmission=1.0,
    ior=1.0,
)


scene.atmosphere_material = atmosphere_material
