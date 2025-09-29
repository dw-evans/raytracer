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

cube_meshes:list[Mesh] = []
for (i, (_f, _material, _csys)) in enumerate(load_data):
    _msh = Mesh(material=_material)
    _triangles = get_triangles_from_obj(f=_f, mesh_idx=_msh.id)
    _msh.csys = _csys


    if _f.stem.lower().startswith("cube."):
        cube_meshes.append(_msh)
    #     x = np.array([x.positions for x in _triangles])
    #     y = x.reshape(-1, 3)
    #     centre = np.mean(y, axis=0)
    #     _triangles = [
    #         scripts.numba_utils.classes.Triangle(
    #             posA = x.posA - centre,
    #             posB = x.posB - centre,
    #             posC = x.posC - centre,
    #             normalA = x.normalA,
    #             normalB = x.normalB,
    #             normalC = x.normalC,
    #             mesh_parent_index = x.mesh_index,
    #             triangle_id = x.id,
    #         )
    #         for x in _triangles
    #     ]
    #     scripts.numba_utils.classes.ALL_TRIANGLES = scripts.numba_utils.classes.ALL_TRIANGLES[:-len(_triangles)] 
    #     _csys = scripts.numba_utils.classes.Csys()
    #     _csys.pos = centre
    #     scripts.numba_utils.classes.add_to_all_triangles(_triangles)
    #     cube_meshes.append(_msh)

    _msh.triangles = _triangles
    scene.meshes.append(_msh)

    timer(chunker.chunk_mesh_bvh)(_msh)


timer(chunker.BVHGraph.register_all_and_update_node_tri_ids())
print(f"There are `{scene.count_triangles()}` triangles in the scene.")

import time
from math import degrees, cos, sin, pi

cube_linear_speeds = []
cube_angular_speeds = []
cube_rotation_axes = []
cube_linear_axes = []
FRAMERATE = 30
for cube_mesh in cube_meshes:
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)  # normalize to unit vector
    omega = np.random.uniform(0, 2*np.pi)

    cube_rotation_axes.append(axis)
    cube_angular_speeds.append(omega)

    speed = 10.0
    cube_linear_speeds.append(speed)
    cube_linear_axes.append(np.array((0, -1, 0), dtype=np.float32))

    pass

FRAMERATE = 60
from .utils import smoothstep, blend

def get_wall_clock_dt(i):
    dt = 1 / FRAMERATE
    if i < 60:
        edge0 = 30
        edge1 = 60
        ss = smoothstep(edge0, edge1, i)
        return blend(ss, 1, 0.001) * dt
    else:
        edge0 = 80
        edge1 = 120
        ss = smoothstep(edge0, edge1, i)
        return blend(ss, 0.001, 0.0) * dt
        
import numpy as np
t = np.arange(120)
v = np.array([get_wall_clock_dt(x) for x in t])
import scipy.integrate
x = scipy.integrate.cumulative_trapezoid(v, t, initial=0)

def get_wall_clock_time(i):
    return x[i]




def animate_cubes(objs:list[Mesh], i, obj_pos_0:list[np.ndarray]):
    # i_reverse = -10
    dt0 = get_wall_clock_time(119)
    # dt = get_wall_clock_time(i) - get_wall_clock_time(i-1)
    for j, obj in enumerate(objs):
        speed = cube_linear_speeds[i]
        axis = cube_linear_axes[i]
        obj.csys.pos = obj_pos_0[i] + np.astype(speed * axis * (get_wall_clock_time(i) - dt0), np.float32)
        # obj.csys.pos += speed * axis * dt
        obj.flag_for_mesh_update()
        pass


def get_frame_number(obj: Scene, i):
    i = i
    if i >= 120:
        return 0
    obj.frame_number = i
    return obj.frame_number



keyframes_fp = import_dir / "camera_keyframes.csv"
import pandas as pd
keyframes_df = pd.read_csv(keyframes_fp)
from math import radians, degrees, sin, cos, pi

import time
import importlib

def animate_camera(obj:Camera, i):
    # row = keyframes_df.iloc[i]
    # obj.csys.set_pos([row["Location Y"], row["Location Z"],-row["Location X"]])
    # obj.csys.quat = np.array([0, 0, 0, 1], dtype=np.float32)
    # obj.csys.rxp(degrees(row["Rotation Y"]))
    # # obj.csys.ryp(degrees(row["Rotation Z"]-pi/2))
    # obj.csys.ryp(0)
    # # obj.csys.rxp(degrees(row["Rotation X"]-pi/2))


    row = keyframes_df.iloc[i]

    obj.csys.set_pos([row["Location Y"], row["Location Z"], -row["Location X"]])
    obj.csys.quat = np.array([0, 0, 0, 1], dtype=np.float32)
    obj.csys.rxp(degrees(row["Rotation Y"]))
    obj.csys.ryp(degrees(row["Rotation Z"]-pi/2))
    obj.csys.rxp(degrees(row["Rotation X"]-pi/2))


    obj.fov = 25
    # obj.csys.pos = np.array((9.4324, 0.005, 0.9541), dtype=np.float32)

    obj.depth_of_field_strength = 0.002
    obj.antialias_strength = 0.002
    obj.near_plane = 9.4
    # obj.near_plane = Vector3(monkey_mesh.csys.pos - obj.csys.pos).squared_length ** 0.5
    obj.bounces_per_ray = 3
    obj.rays_per_pixel = 2
    obj.passes_per_frame = 20
    obj.chunksx = 2
    obj.chunksy = 2
    # time.sleep(1.0)

    pass



from functools import partial
scene.animations.append(FrameAnimation(scene, get_frame_number))
# scene.animations.append(FrameAnimation(scene.cam, animate_camera_params))
scene.animations.append(FrameAnimation(scene.cam, animate_camera))
scene.animations.append(FrameAnimation(cube_meshes, partial(animate_cubes, obj_pos_0=[x.csys.pos for x in cube_meshes])))


animate_camera(scene.cam, 0)

