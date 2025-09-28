import scripts.numba_utils.classes
import trimesh
from . import chunker
import numpy as np
import typing
from scripts.classes import Mesh

if typing.TYPE_CHECKING:
    from scripts.classes import Triangle

def get_triangles_from_obj(f, mesh_idx) -> list["Triangle"]:
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


def reset_global_datas():
    scripts.numba_utils.classes.reset_all_triangles()
    Mesh.reset()
    chunker.BVHGraph.reset()
    chunker.BVHParentNode.reset()