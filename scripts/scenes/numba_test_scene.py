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
    Vector4((0.2, 0.0, 0.4, 1.0), dtype="f4"),
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

pass
# msh1 = Mesh.from_obj(
#     Path() / "objects/smooth_disc.obj",
#     material=glass_material,
# )

# msh1 = Mesh.from_obj(
#     Path() / "objects/tyre_tread.obj",
#     material=glass_material,
# )

import trimesh
import numpy as np
import numba_scripts.classes

# msh = trimesh.load("objects/tyre_tread.obj")

# vertex_indices_arr = msh.faces.astype(np.int32)
# vertices = msh.vertices.astype(np.float32)
# vertex_normals = msh.vertex_normals.astype(np.float32)


# triangles = numba_scripts.classes.triangles_from_obj_data(
#     vertex_indices_arr,
#     vertices,
#     vertex_normals,
# )

# msh1 = Mesh(material=glass_material)
# msh1.csys = numba_scripts.classes.Csys()
# msh1.triangles = triangles
# msh1.csys.tg(np.array([0, 0.0, 4.0]))
# scene.meshes.append(msh1)

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

body_material = Material(
    Vector4([0.674, 0.203, 0.600, 1.0]),
    smoothness = 0.95,
)

chrome_material = Material(
    Vector4([0.29, 0.29, 0.29, 1.0]),
    smoothness = 0.9,
)

glass_material = Material(
    Vector4([1., 1, 1, 1]),
    transmission = 1.0,
    smoothness=1.0,
    ior = 1.5,
)
roof_material = Material(
    Vector4([0.095, 0.01, 0.016, 1.0]),
    smoothness=0,
)
hubcap_material = Material(
    Vector4([0.8, 0.8, 0.8, 1.0]),
    smoothness = 0.5,
)

black_material = Material(
    Vector4([0.1, 0.1, 0.1, 1.0]),
    smoothness = 0.5,
)

rubber_material = Material(
    Vector4([0.1, 0.1, 0.1, 1.0]),
    smoothness = 0.5,
)

light_material_1 = Material(
    Vector4([1., 0.327, 0.0, 1.0]),
    transmission = 0.8,
    smoothness=1.0,
    ior = 1.1,
)
light_material_2 = Material(
    Vector4([1., 0.0, 0.0, 1.0]),
    transmission = 0.8,
    smoothness=1.0,
    ior = 1.1,
)

internal_light_material = Material(
    Vector4([0.0, 0.0, 0.0, 1.0]),
    emissionColor=Vector3([1., 1., 1.]),
    emissionStrength=5.0
)

internal_plastic_material = Material(
    Vector4([0.3, 0.3, 0.3, 1.0]),
    smoothness = 0.5,
)

internal_white_material = Material(
    Vector4([0.1, 0.1, 0.1, 1.0]),
    smoothness = 0.5,
)

seatbelt_material = Material(
    Vector4([0.15, 0.15, 0.15, 1.0]),
    smoothness = 0.5,
)

meshes = []

load_data = [
    # ("objects/car/black.obj", glass_material),
    # ("objects/car/body1.obj", body_material),
    # ("objects/car/chrome.obj", chrome_material),
    ("objects/car/glass1.obj", glass_material),
    # ("objects/car/internal_light.obj", internal_light_material),
    # ("objects/car/internal_plastic.obj", internal_plastic_material),
    # ("objects/car/internal_white.obj", internal_white_material),
    # ("objects/car/lights_rear_bottom.obj", light_material_1),
    # ("objects/car/lights_rear_top.obj", light_material_2),
    # ("objects/car/roof_fabric.obj", roof_material),
    # ("objects/car/rubber.obj", rubber_material),
    # ("objects/car/seatbelts.obj", seatbelt_material),
    # ("objects/car/wheels.obj", hubcap_material),

    # ("objects/car/cube_test.obj", glass_material),
    # ("objects/car/cube_test_2.obj", glass_material),
    # ("objects/car/cube_test_3.obj", glass_material),
    # ("objects/smooth_disc.obj", glass_material),

    ("objects/car/glass_test2.obj", glass_material),
    ("objects/car/glass_test3.obj", glass_material),
]
# ]

car_csys = numba_scripts.classes.Csys()
car_csys.pos = np.array([0.0, -1.0, 4.0], dtype=np.float32)
car_csys.ryg(180-45)

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

# from ..animate import Animation
# from math import sin, cos, tan, pi

# import numba_scripts.classes
# def mesh_csys_animate(obj: numba_scripts.classes.Csys, t):
#     obj.pos[0] = 0.0 + 10.0 * sin(t / 2)
#     obj.pos[1] = -0.5
#     obj.pos[2] = 6.0 + 5.0 * sin(t / 3)


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
