from ..classes import (
    Scene,
    Mesh,
    Triangle,
    Sphere,
    Material,
    Csys,
    Camera,
)
from ..animate import Animation, FrameAnimation

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
    smoothness=0.0,
)
material_blue = Material(
    Vector4((0.0, 0.0, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.0,
)

material_rear_wall_animated = Material()
material_front_wall_animated = Material()

material_green = Material(
    Vector4((0.0, 1.0, 0.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.0,
)
material_white_upper = Material(
    Vector4((1.0, 1.0, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.0,
)
material_white_lower = Material(
    Vector4((1.0, 1.0, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.0,
)

material_red_passthrough = Material(
    Vector4((1.0, 0.0, 0.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.0,
    transparent_from_behind=True
)
material_blue_passthrough = Material(
    Vector4((0.0, 0.0, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.0,
    transparent_from_behind=True
)
material_green_passthrough = Material(
    Vector4((0.0, 1.0, 0.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.0,
    transparent_from_behind=True
)
material_white_passthrough = Material(
    Vector4((1.0, 1.0, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.0,
    ior=1.0,
    transparent_from_behind=True
)

material_light_source_1 = Material(
    Vector4((0.0, 0.0, 0.0, 1.0), dtype="f4"),
    emissionColor=Vector3((1, 1.0, 1), dtype="f4"),
    emissionStrength=2.0,
)

glass_material = Material(
    color=Vector4((1.0, 0.9, 0.9, 1.0)),
    smoothness=1.0,
    transmission=0.0,
    ior=1.6,
)

atmosphere_material = Material(
    Vector4((0.5, 0.5, 0.5, 1.0),  dtype="f4"),
    transmission=1.0,
    ior=1.0,
)

scene = Scene()
scene.atmosphere_material = atmosphere_material

meshes = []


# material_subject = Material(
#     color=Vector4((1.0, 0.3, 0.6, 1.0)),
#     smoothness=1.0,
#     transmission=0.0,
#     ior=1.6,
# )

material_red_1 = Material(
    Vector4((1.0, 1.0, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=1.0,
    specularStrength=0.2,
    ior=1.0,
)

csys0 = numba_scripts.classes.Csys()

monkey_file="objects/monkey.obj"
load_data = [
    # ("objects/old/final_scene/wall_left.obj", material_red, csys0),
    # ("objects/old/final_scene/wall_right.obj", material_green, csys0),
    # ("objects/old/final_scene/wall_top.obj", material_white_upper, csys0),
    ("objects/old/final_scene/wall_bottom.obj", material_white_lower, csys0),
    # (r"objects\monkey.blend\wall_bottom.obj", material_white_lower, csys0),
    # ("objects/old/final_scene/wall_front.obj", material_front_wall_animated, csys0),
    # ("objects/old/final_scene/wall_back.obj", material_rear_wall_animated, csys0),
    # ("objects/old/final_scene/light.obj", material_light_source_1, csys0),
    # ("objects/old/final_scene/subject.obj", material_subject),
    # (monkey_file:="objects/monkey.obj", material_red_1, numba_scripts.classes.Csys()),
    (monkey_file:=monkey_file, material_red_1, Csys()),
]


# car_csys._pos = np.array([0.0, 1.0, 8.0], dtype=np.float32)
csys0._pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
csys0.ryg(180)

spheres = [
    # Sphere(
    #     pos=car_csys._pos + np.array([0.0, 1.5, 0.0]),
    #     radius=1.0,
    #     material=material_red_1,
    # ),
]

scene.spheres = spheres

# scene.cam.csys._pos = np.array([0.0, 2.0, 0.0], dtype=np.float32)

for (i, (f, material, csys)) in enumerate(load_data):
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

    msh1.csys = csys
    if f == monkey_file:
        monkey_mesh = msh1

    msh1.triangles = triangles

    scene.meshes.append(msh1)





print(f"There are `{scene.count_triangles()}` triangles in the scene.")

# from ..animate import Animation
# from math import sin, cos, tan, pi

# import numba_scripts.classes

# def mesh_csys_animate(obj: numba_scripts.classes.Csys, t, x0:Vector3):
#     obj._pos[0] = x0[0] + 1.0 * sin(t / 2)
#     obj._pos[1] = x0[1]
#     obj._pos[2] = x0[2] + 1.0 * sin(t / 3)
#     print("here")

# from functools import partial
# scene.animations.append(Animation(scene.meshes[-1].csys, partial(mesh_csys_animate, x0=scene.meshes[0].csys._pos)))

START_CHUNK_SIZE_FRAC = 1.0

# base_material = Material(
#     Vector4((1.0, 1.0, 1.0, 1.0), dtype="f4"),
#     Vector3((0.0, 0.0, 0.0), dtype="f4"),
#     smoothness=1.0,
#     specularStrength=0.2,
#     ior=1.0,
# )
# fname = "objects/car_new/rubber.obj"

# from .chunker import load_chunked_mesh_into_scene

# r = load_chunked_mesh_into_scene(scene, fname, base_material, car_csys)


keyframes_fp = Path() / "blender/camera_keyframes.csv"
import pandas as pd
keyframes_df = pd.read_csv(keyframes_fp)
from math import radians, degrees, sin, cos, pi

time_injectino_modified = 0.0
import time
import importlib
from . import injection
def animate_camera(obj:Camera, i):
    row = keyframes_df.iloc[i]

    obj.csys.set_pos([row["Location Y"], row["Location Z"], -row["Location X"]])
    obj.csys.quat = np.array([0, 0, 0, 1], dtype=np.float32)
    obj.csys.rxp(degrees(row["Rotation Y"]))
    obj.csys.ryp(degrees(row["Rotation Z"]-pi/2))
    obj.csys.rxp(degrees(row["Rotation X"]-pi/2))


def animate_monkey(obj:Mesh, i):
    obj.csys.set_pos([0.0, 0.0, 0.0])
    obj.csys._pos = np.array([0.0, 1.0 + 0.2*sin(2*pi * i/60), 0.0], dtype=np.float32)
    obj.csys.quat = np.array([0, 0, 0, 1], dtype=np.float32)
    obj.csys.ryg(0 + 180 + 5 *i)
    obj.csys.rxp(30)
    # t = triangles[0]
    # p0 = t.positions
    # numba_scripts.classes.update_triangles_to_csys(obj.triangles, obj.csys)
    # p1 = t.positions

    pass
    



def smoothstep(edge0, edge1, x):
    # normalize x to [0, 1]
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def animate_rear_material(obj:Material, i):
    edge0 = 0
    edge1 = 30
    mat1 = Material(
        Vector4((0.0, 0.0, 1.0, 1.0), dtype="f4"),
        Vector3((0.0, 0.0, 0.0), dtype="f4"),
        smoothness=0.0,
    )
    mat2 = Material(
        Vector4((1.0, 1.0, 1.0, 1.0), dtype="f4"),
        Vector3((0.0, 0.0, 0.0), dtype="f4"),
        smoothness=1.0,
        specularStrength=1.0
    )

    ss = smoothstep(edge0, edge1, i)
    
    obj.smoothness = (1-ss) * mat1.smoothness + ss * mat2.smoothness
    obj.color = (1-ss) * mat1.color + ss * mat2.color
    obj.specularStrength = (1-ss) * mat1.specularStrength + ss * mat2.specularStrength
    
    # obj.transmission = ss

    pass

_i_end:int = None
_mat1:Material = None
_mat2:Material = None

def animate_front_material(obj:Material, i):
    global _i_end
    global _mat1
    global _mat2

    obj.transparent_from_behind = True
    transmission = 0.0

    if i < 30:
        # No wall to a blue wall
        edge0 = 25
        edge1 = 30
        ss = smoothstep(edge0, edge1, i)
        obj.color = Vector4((1.0, 1.0, 1.0, 1.0))
        obj.transparent_from_behind = True

        _mat1 = Material(
            Vector4((0.0, 0.0, 1.0, 0.0), dtype="f4"),
            Vector3((0.0, 0.0, 0.0), dtype="f4"),
        )
        _mat2 = Material(
            Vector4((0.0, 0.0, 0.8, 1.0), dtype="f4"),
            Vector3((0.0, 0.0, 0.0), dtype="f4"),
            smoothness=0.0,
        )
        obj.color = (1-ss) * _mat1.color + ss * _mat2.color


    # elif i < 150:
    else:
        # blue wall to a mirror
        edge0 = 45
        edge1 = 60
        ss = smoothstep(edge0, edge1, i)

        _mat1 = Material(
            Vector4((0.0, 0.0, 0.8, 1.0), dtype="f4"),
            Vector3((0.0, 0.0, 0.0), dtype="f4"),
            smoothness=0.0,
        )
        _mat2 = Material(
            Vector4((1.0, 1.0, 1.0, 1.0), dtype="f4"),
            Vector3((0.0, 0.0, 0.0), dtype="f4"),
            smoothness=1.0,
            specularStrength=1.0
        )
        obj.color = (1-ss) * _mat1.color + ss * _mat2.color
        obj.smoothness = (1-ss) * _mat1.smoothness + ss * _mat2.smoothness
        obj.specularStrength = (1-ss) * _mat1.specularStrength + ss * _mat2.specularStrength

        pass


def animate_internal_materials_specular_partial(obj:Material, i:int, edge0:int, edge1:int, mat0:Material):
    ss = smoothstep(edge0, edge1, i)
    # obj.color = (1-ss) * mat0.color + ss * _mat2.color
    obj.specularStrength = (1-ss) * mat0.specularStrength + (1.0-mat0.specularStrength) * ss
    obj.specularColor = ((1-ss) * Vector3((1.0, 1.0, 1.0)) + ss * Vector3(mat0.color.xyz)).normalized
    obj.smoothness = (1-ss) * mat0.smoothness + (0.99-mat0.smoothness) * ss
    pass

def animate_camera_params(obj:Camera, i:int):
    # obj.depth_of_field_strength = 0.02
    # obj.depth_of_field_strength = 0.1
    obj.depth_of_field_strength = 0.000
    obj.antialias_strength = 0.001
    # obj.near_plane = 8.5
    obj.near_plane = Vector3(monkey_mesh.csys.pos - obj.csys.pos).squared_length ** 0.5
    obj.bounces_per_ray = 4
    obj.rays_per_pixel = 8
    obj.passes_per_frame = 100
    obj.chunksx = 5
    obj.chunksy = 5

    obj.csys.pos = (2* monkey_mesh.csys.pos + obj.csys.pos) /3.0
    
    # add chunking
    # def check():
    #     (obj.bounces_per_ray * obj.rays_per_pixel * obj.passes_per_frame / obj.chunksx / obj.chunksy) > 8

    # while not check():
    #     obj.rays_per_pixel -= 1

    # obj.aspect
    # obj.fov

def get_frame_number(obj: Scene, i):
    i = i
    # i = 0
    obj.frame_number = i
    return obj.frame_number


import copy
from functools import partial
scene.animations.append(FrameAnimation(scene, get_frame_number))

scene.animations.append(FrameAnimation(scene.cam, animate_camera))
# scene.animations.append(FrameAnimation(monkey_mesh, animate_monkey))
scene.animations.append(FrameAnimation(material_rear_wall_animated, animate_rear_material))
scene.animations.append(FrameAnimation(material_front_wall_animated, animate_front_material))
scene.animations.append(FrameAnimation(material_red, partial(animate_internal_materials_specular_partial, edge0=20, edge1=45, mat0=copy.deepcopy(material_red))))
scene.animations.append(FrameAnimation(material_green, partial(animate_internal_materials_specular_partial, edge0=25, edge1=50, mat0=copy.deepcopy(material_green))))
scene.animations.append(FrameAnimation(material_white_upper, partial(animate_internal_materials_specular_partial, edge0=25, edge1=50, mat0=copy.deepcopy(material_white_upper))))
scene.animations.append(FrameAnimation(material_white_lower, partial(animate_internal_materials_specular_partial, edge0=30, edge1=55, mat0=copy.deepcopy(material_white_lower))))
scene.animations.append(FrameAnimation(scene.cam, animate_camera_params))


# monkey_mesh = scene.meshes[0]

animate_camera(scene.cam, 0)
# animate_monkey(monkey_mesh, 0)
animate_rear_material(material_rear_wall_animated, 0)
animate_front_material(material_front_wall_animated, 0)
animate_front_material(scene.cam, 0)




from . import chunker 
from numba_scripts.functions import timer

import importlib
importlib.reload(chunker)


# chunk_mesh_bvh(monkey_mesh)
# chunk_mesh_bvh(scene.meshes[3])
chunker.BVHGraph.reset()
# chunker.chunk_mesh_bvh(scene.meshes[0])
timer(chunker.chunk_mesh_bvh)(scene.meshes[1])
# chunk_mesh_bvh(scene.meshes[-5])

import itertools

scene.reset_and_register_triangles()

timer(chunker.BVHGraph.register_all)()

pass
import pandas as pd

# pd.DataFrame([list(itertools.chain.from_iterable(x.aabb.tolist())) + [x.is_leaf(), len(x.tris)] for x in chunker.BVHGraph.GRAPHS[0].leaf_nodes])
