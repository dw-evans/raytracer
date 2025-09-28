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



material_light_source_1 = Material(
    Vector4((0.0, 0.0, 0.0, 1.0), dtype="f4"),
    emissionColor=Vector3((1, 1.0, 1), dtype="f4"),
    emissionStrength=5.0,
)

atmosphere_material = Material(
    Vector4((0.5, 0.5, 0.5, 1.0),  dtype="f4"),
    transmission=1.0,
    ior=1.0,
)


material_floor = Material(
    Vector4((.8, .8, .8, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.0,
)

material_dragon = Material(
    Vector4((.722, .457, 1.0, 1.0), dtype="f4"),
    Vector3((0.0, 0.0, 0.0), dtype="f4"),
    smoothness=0.5,
    specularStrength=0.2,
    ior=1.0,
)

scene = Scene()
scene.atmosphere_material = atmosphere_material

csys0 = scripts.numba_utils.classes.Csys()


import random
import_dir = Path() / "objects/monkey_blend/dragon"
load_data = []


load_data.append((import_dir / "dragon-3.obj", material_dragon, csys0))
load_data.append((import_dir / "light.obj", material_light_source_1, csys0))
load_data.append((import_dir / "floor.obj", material_floor, csys0))
load_data.append((import_dir / "base.obj", material_floor, csys0))


def random_pastel():
    # Generate RGB values between 0.7 and 1.0 for soft, bright colors
    r = random.uniform(0.7, 1.0)
    g = random.uniform(0.7, 1.0)
    b = random.uniform(0.7, 1.0)
    return (r, g, b)

random.seed(0)
colours = [random_pastel() for _ in range(6)]

# load cubes
for i, fp in enumerate(import_dir.glob("*.obj")):
    if fp.stem.startswith("Cube."):
        material = Material(
            Vector4((*random.choice(colours), 1.0), dtype="f4"),
            Vector3((0.0, 0.0, 0.0), dtype="f4"),
            smoothness=0.0,
        )
        load_data.append((fp, material, csys0))


csys0._pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
csys0.ryg(-90)


from .utils import get_triangles_from_obj, reset_global_datas


# Reset all triangles, meshes, graphs and nodes
reset_global_datas()

for (i, (_f, _material, _csys)) in enumerate(load_data):
    _msh = Mesh(material=_material)
    _triangles = get_triangles_from_obj(f=_f, mesh_idx=_msh.id)
    _msh.csys = _csys

    _msh.triangles = _triangles
    scene.meshes.append(_msh)

    timer(chunker.chunk_mesh_bvh)(_msh)


timer(chunker.BVHGraph.register_all_and_update_node_tri_ids())
print(f"There are `{scene.count_triangles()}` triangles in the scene.")

import time
from math import degrees, cos, sin, pi
def animate_camera_params(obj:Camera, i:int):

    pos = [9.432361602783203, 0.059754848,	0.9540680646896362]
    rot = [1.5707963705062866,	0,	1.5707963705062866]

    obj.csys.set_pos([pos[1], pos[2], -pos[0]])
    obj.csys.quat = np.array([0, 0, 0, 1], dtype=np.float32)
    obj.csys.rxp(degrees(rot[1]))
    obj.csys.ryp(degrees(rot[2]-pi/2))
    obj.csys.rxp(degrees(rot[0]-pi/2))

    obj.fov = 20
    # obj.csys.pos = np.array((9.4324, 0.005, 0.9541), dtype=np.float32)

    obj.depth_of_field_strength = 0.00
    obj.antialias_strength = 0.002
    # obj.near_plane = 8.5
    # obj.near_plane = Vector3(monkey_mesh.csys.pos - obj.csys.pos).squared_length ** 0.5
    obj.bounces_per_ray = 8
    obj.rays_per_pixel = 1
    obj.passes_per_frame = 10000
    obj.chunksx = 2
    obj.chunksy = 2
    # time.sleep(1.0)


def get_frame_number(obj: Scene, i):
    i = 0
    obj.frame_number = i
    return obj.frame_number

scene.animations.append(FrameAnimation(scene.cam, animate_camera_params))

