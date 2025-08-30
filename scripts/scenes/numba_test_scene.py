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
from numba_scripts.classes import Csys

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
            0.2,
            0.2,
            0.2,
            1.0,
        ),
        dtype="f4",
    ),
    transmission=1.0,
    ior=1.0,
)

scene.atmosphere_material = atmosphere_material

body_material = Material(
    # Vector4([0.674, 0.203, 0.600, 1.0]),
    Vector4([0.6706, 0.7373, 0.8392, 1.0]),
    smoothness=0.8,
)



chrome_material = Material(
    Vector4([0.29, 0.29, 0.29, 1.0]),
    smoothness=0.95,
)

glass_material = Material(
    Vector4([1.0, 1, 1, 1]),
    transmission=1.0,
    smoothness=1.0,
    ior=1.5,
)
roof_material = Material(
    Vector4([0.3412, 0.1020, 0.1333, 1.0]),
    smoothness=0,
)
hubcap_material = Material(
    Vector4([0.8, 0.8, 0.8, 1.0]),
    smoothness=0.5,
)

black_material = Material(
    Vector4([0.1, 0.1, 0.1, 1.0]),
    smoothness=0.5,
)

rubber_material = Material(
    Vector4([0.1, 0.1, 0.1, 1.0]),
    smoothness=0.5,
)

light_material_1 = Material(
    Vector4([1.0, 0.327, 0.0, 1.0]),
    transmission=0.8,
    smoothness=1.0,
    ior=1.6,
)
light_material_2 = Material(
    Vector4([1.0, 0.0, 0.0, 1.0]),
    transmission=0.8,
    smoothness=1.0,
    ior=1.6,
)

internal_light_material = Material(
    Vector4([0.0, 0.0, 0.0, 1.0]),
    emissionColor=Vector3([1.0, 1.0, 1.0]),
    emissionStrength=5.0,
)

internal_plastic_material = Material(
    Vector4([0.3, 0.3, 0.3, 1.0]),
    smoothness=0.5,
)

internal_white_material = Material(
    Vector4([0.95, 0.95, 0.95, 1.0]),
    smoothness=0.5,
)

seatbelt_material = Material(
    Vector4([0.15, 0.15, 0.15, 1.0]),
    smoothness=0.5,
)

floor_material = Material(
    Vector4((0.5, 0.5, 0.5, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    0.0,
    smoothness=0.0,
)

light_bar_material = Material(
    Vector4([0.0, 0.0, 0.0, 1.0]),
    emissionColor=Vector3([1.0, 1.0, 1.0]),
    emissionStrength=3.0,
)
light_bar_cover_material = Material(
    Vector4((0.1, 0.1, 0.1, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    0.0,
    smoothness=0.0,
)

meshes = []


"abbcd6" # car color blue
"571a22" # cloth color

DEFAULT_CSYS = Csys()

scene.cam.csys.pos = pyrr.Vector3([0, 1.0, 0], dtype="f2")

car_csys = numba_scripts.classes.Csys()
car_csys.set_pos([0.0, 0.0, 8.0])
car_csys.ryg(180-45)
# car_csys.ryg(180)

name_material_pose = [
    # ("objects/testing/test-cube.obj", rubber_material, Csys().set_pos((0, 0, 0))),
    # ("objects/testing/window-left.obj", glass_material),
    # ("objects/testing/window-right.obj", glass_material),
    # ("objects/testing/window-front.obj", glass_material),
    # ("objects/testing/test-window-2.obj", glass_material),
    # ("objects/testing/test-window-7.obj", glass_material),
    # ("objects/car/glass_test5.obj", glass_material, Csys().set_pos((0, 0, 8))),
    # ("objects/car/black.obj", black_material, car_csys),
    # ("objects/car/body1.obj", body_material, car_csys),

    ("objects/car_new/body-chrome.obj", chrome_material, car_csys),
    ("objects/car_new/body-panels.obj", body_material, car_csys),
    ("objects/car_new/floor.obj", floor_material, car_csys),
    ("objects/car_new/foglights-glass.obj", glass_material, car_csys),
    ("objects/car_new/glass.obj", glass_material, car_csys),
    ("objects/car_new/headlights-glass.obj", glass_material, car_csys),
    ("objects/car_new/light-inner.obj", light_bar_material, car_csys),
    ("objects/car_new/light-outer.obj", light_bar_cover_material, car_csys),
    ("objects/car_new/roof.obj", roof_material, car_csys),
    ("objects/car_new/rubber.obj", rubber_material, car_csys),
    ("objects/car_new/wheels.obj", hubcap_material, car_csys),

]




for i, (_fname, _material, _csys) in enumerate(name_material_pose):
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
