from scripts.classes import (
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
import scripts.numba_utils.classes

from . import chunker 
from scripts.numba_utils.functions import timer


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

material_subject = Material(
    Vector4((1.0, 1.0, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=1.0,
    specularStrength=0.2,
    ior=1.0,
)

material_floor = Material(
    Vector4((1.0, 1.0, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.2,
    specularStrength=0.2,
)

csys0 = scripts.numba_utils.classes.Csys()
csys_dragon = scripts.numba_utils.classes.Csys()
csys_dragon.ryg(0)

# monkey_file="objects/monkey_blend2/monkey.obj"
monkey_file="objects/monkey_blend/monkey.obj"

load_data = [
    # (monkey_file, material_subject, Csys()),
    ("objects/monkey_blend/dragon.obj", material_subject, csys_dragon),
    ("objects/monkey_blend/wall_top.obj", material_white_upper, csys0),
    ("objects/monkey_blend/wall_bottom.obj", material_white_lower, csys0),
    ("objects/monkey_blend/wall_left.obj", material_red, csys0),
    ("objects/monkey_blend/wall_right.obj", material_green, csys0),
    ("objects/monkey_blend/wall_front.obj", material_front_wall_animated, csys0),
    ("objects/monkey_blend/wall_back.obj", material_rear_wall_animated, csys0),
    ("objects/monkey_blend/light.obj", material_light_source_1, csys0),
    ("objects/monkey_blend/ground.obj", material_floor, csys0),
]


csys0._pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
csys0.ryg(-90)



import importlib
importlib.reload(chunker)

wd = Path(__file__).parent.parent.parent

def get_triangles_from_obj(f, mesh_idx) -> list[Triangle]:
    msh = trimesh.load(f)

    vertex_indices_arr = msh.faces.astype(np.int32)
    vertices = msh.vertices.astype(np.float32)
    vertex_normals = msh.vertex_normals.astype(np.float32)
    
    start_offset = len(scripts.numba_utils.classes.get_all_triangles_arr())
    triangles = scripts.numba_utils.classes.create_and_register_triangles_from_obj_data(
        vertex_indices_arr,
        vertices,
        vertex_normals,
        mesh_idx,
        start_offset,
    )
    if not triangles:
        raise Exception
    # append the triangles to the all triangles list
    scripts.numba_utils.classes.add_to_all_triangles(triangles)
    return triangles


# Reset all triangles, meshes, graphs and nodes
scripts.numba_utils.classes.reset_all_triangles()
Mesh.reset()
chunker.BVHGraph.reset()
chunker.BVHParentNode.reset()

pass

for (i, (_f, _material, _csys)) in enumerate(load_data):

    _msh = Mesh(material=_material)
    _triangles = get_triangles_from_obj(f=_f, mesh_idx=_msh.id)


    _msh.csys = _csys
    if _f == monkey_file:
        monkey_mesh = _msh

    _msh.triangles = _triangles
    scene.meshes.append(_msh)

    timer(chunker.chunk_mesh_bvh)(_msh)

    pass

# if globals().get("monkey_file", None):
#     print("warning bodged monkey mesh!")
#     monkey_file = _msh

# Update the scene's triangles (and correct the triangles internal id...)
# scene.reset_and_register_triangles_and_update_their_ids()
# Update the graphs leaf triangle struct, and update the node ids
timer(chunker.BVHGraph.register_all_and_update_node_tri_ids())

print(f"There are `{scene.count_triangles()}` triangles in the scene.")

keyframes_fp = Path() / "input/camera_keyframes.csv"
import pandas as pd
keyframes_df = pd.read_csv(keyframes_fp)
from math import radians, degrees, sin, cos, pi

time_injectino_modified = 0.0
import time
import importlib

def animate_camera(obj:Camera, i):
    row = keyframes_df.iloc[i]

    obj.csys.set_pos([row["Location Y"], row["Location Z"], -row["Location X"]])
    obj.csys.quat = np.array([0, 0, 0, 1], dtype=np.float32)
    obj.csys.rxp(degrees(row["Rotation Y"]))
    obj.csys.ryp(degrees(row["Rotation Z"]-pi/2))
    obj.csys.rxp(degrees(row["Rotation X"]-pi/2))


def animate_monkey(obj:Mesh, i):
    # i=0
    # obj.csys.set_pos([0.0, 3.0, 0.0])
    obj.csys._pos = np.array([0.0, 1.0 + 0.2*sin(2*pi * i/60), 0.0], dtype=np.float32)
    obj.csys.quat = np.array([0, 0, 0, 1], dtype=np.float32)
    obj.csys.ryg(0 + 180 + 5 *i)
    obj.csys.rxp(30)
    obj.flag_for_mesh_update()
    # t = triangles[0]
    # p0 = t.positions
    # scripts.numba_utils.classes.update_triangles_to_csys(obj.triangles, obj.csys)
    # p1 = t.positions

    pass
    

def smoothstep(edge0, edge1, x):
    # normalize x to [0, 1]
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def animate_rear_material(obj:Material, i):
    edge0 = 20
    edge1 = 50
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

    edge1a = 80
    if i < edge1a:
        # No wall to a blue wall
        edge0 = edge1a - 30
        edge1 = edge1a
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
        edge0 = max(60, edge1a)
        edge1 = edge0 + 15
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
    obj.depth_of_field_strength = 0.00
    obj.antialias_strength = 0.001
    # obj.near_plane = 8.5
    # obj.fov = 39.6
    obj.fov = 15
    if globals().get("monkey_mesh", None):
        obj.near_plane = Vector3(monkey_mesh.csys.pos - obj.csys.pos).squared_length ** 0.5
        obj.csys.pos = (1.0* monkey_mesh.csys.pos + obj.csys.pos) /2.0
    # obj.near_plane = Vector3(monkey_mesh.csys.pos - obj.csys.pos).squared_length ** 0.5
    obj.bounces_per_ray = 6
    obj.rays_per_pixel = 1
    obj.passes_per_frame = 10000
    obj.chunksx = 2
    obj.chunksy = 2

    
    # add chunking
    # def check():
    #     (obj.bounces_per_ray * obj.rays_per_pixel * obj.passes_per_frame / obj.chunksx / obj.chunksy) > 8

    # while not check():
    #     obj.rays_per_pixel -= 1

    # obj.aspect
    # obj.fov

def blend(factor, a, b):
    return factor * a + (1-factor) * b

def animate_light_material(obj:Material, i, initial_material:Material):
    edge0 = 50
    edge1 = edge0 + 20
    ss = smoothstep(edge0, edge1, i)
    obj.emissionStrength = (1-ss) * initial_material.emissionStrength + ss * initial_material.emissionStrength * 5.0


def get_frame_number(obj: Scene, i):
    # i = i
    i = 120
    obj.frame_number = i
    return obj.frame_number



import copy
from functools import partial
scene.animations.append(FrameAnimation(scene, get_frame_number))

"""
keyframes

"""

scene.animations.append(FrameAnimation(scene.cam, animate_camera))
# scene.animations.append(FrameAnimation(monkey_mesh, animate_monkey))
scene.animations.append(FrameAnimation(material_rear_wall_animated, animate_rear_material))
scene.animations.append(FrameAnimation(material_front_wall_animated, animate_front_material))
scene.animations.append(FrameAnimation(material_red, partial(animate_internal_materials_specular_partial, edge0=120, edge1=150, mat0=copy.deepcopy(material_red))))
scene.animations.append(FrameAnimation(material_green, partial(animate_internal_materials_specular_partial, edge0=150, edge1=180, mat0=copy.deepcopy(material_green))))
scene.animations.append(FrameAnimation(material_white_upper, partial(animate_internal_materials_specular_partial, edge0=90, edge1=120, mat0=copy.deepcopy(material_white_upper))))
scene.animations.append(FrameAnimation(material_white_lower, partial(animate_internal_materials_specular_partial, edge0=60, edge1=90, mat0=copy.deepcopy(material_white_lower))))
scene.animations.append(FrameAnimation(scene.cam, animate_camera_params))
scene.animations.append(FrameAnimation(material_light_source_1, partial(animate_light_material, initial_material=copy.deepcopy(material_light_source_1))))


animate_camera(scene.cam, 0)
# animate_monkey(monkey_mesh, 0)
animate_rear_material(material_rear_wall_animated, 0)
animate_front_material(material_front_wall_animated, 0)
animate_front_material(scene.cam, 0)



# pass
# import pandas as pd

# pd.DataFrame([list(itertools.chain.from_iterable(x.aabb.tolist())) + [x.is_leaf(), len(x.tris)] for x in chunker.BVHGraph.GRAPHS[0].leaf_nodes])

