
from __future__ import annotations
import trimesh
from ..classes import Material, Scene, Mesh
import pyrr
import numpy as np
import numba_scripts.classes

from numba_scripts.classes import Csys
import struct


# def load_mesh_into_scene(_scene:Scene, _fname:str, _material:Material, _csys:Csys, start_chunk_size_frac:float=0.5):


def load_chunked_mesh_into_scene(_scene:Scene, _fname:str, _material:Material, _csys:Csys, start_chunk_size_frac:float=0.5):
    _target_tris_per_trunk = 10000
    _decimation_factor = 0.0


    msh:trimesh.Trimesh = trimesh.load(_fname)

    aabb = pyrr.aabb.create_from_points(msh.vertices)
    centroids = msh.triangles_center  # shape (n_triangles, 3)

    centroids = msh.triangles_center

    longest_axis = np.argmax(abs(aabb[0]-aabb[1]))
    msh_origin = aabb[0]
    normalized_centroids = centroids - msh_origin
    # chunk_size_frac = START_CHUNK_SIZE_FRAC
    chunk_size_frac = start_chunk_size_frac

    do_accept = False
    max_attempts = 10
    attempts = 1
    while not do_accept:
        if attempts > max_attempts:
            print(f"Warning, unable to resolve chunking in {max_attempts} attempts, exiting")
            break

        chunk_size = abs(aabb[0]-aabb[1])[longest_axis] * chunk_size_frac

        grid_coords = np.floor(normalized_centroids / chunk_size).astype(int)

        values = range(round(1 / chunk_size_frac + 0.5+1))
        x, y, z = np.meshgrid(values, values, values, indexing='ij')
        combinations = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        matches = (grid_coords[:, None, :] == combinations[None, :, :]).all(axis=-1)  

        # Sum over points to get counts per combination
        counts = matches.sum(axis=0)
        print(f"maximum tri count per chunk = {counts.max()}")

        if counts.max() > _target_tris_per_trunk:
            fac = max((counts.max()/_target_tris_per_trunk) ** 0.33, 2.0)
            print(f"next reduction factor = {fac}")
            chunk_size_frac /= fac

            attempts += 1
            continue

        do_accept = True


        tri_to_chunk_id_arr = np.argmax(matches, axis=1)
        unique_chunk_ids, _ = np.unique(tri_to_chunk_id_arr, return_inverse=True)

        print(f"nonzero chunks count = {np.sum(counts > 0)}")
        pass

    vertex_indices_arr = msh.faces.astype(np.int32)
    vertices = msh.vertices.astype(np.float32)
    vertex_normals = msh.vertex_normals.astype(np.float32)



    results = []

    for i, chunk_id in enumerate(unique_chunk_ids):
        # indices
        # the indices where the chunk id matches the required


        indices, = np.where(tri_to_chunk_id_arr == chunk_id)

        if _decimation_factor > 0.0:
            indices = indices[np.random.choice(len(indices), size=max(int(((1-_decimation_factor) * len(indices))), 1), replace=False)]

        # calculate the facets within the chunk, remap the indices to account for the trimmed items.
        facets_within_chunk_original = vertex_indices_arr[indices]
        z = facets_within_chunk_original.reshape(-1)
        map_from = np.unique(z)
        map_to = np.arange(map_from.shape[0])

        new_indices = np.searchsorted(map_from, z)
        facet_vertex_indices_within_chunk = new_indices.reshape(-1, 3)

        # calculate the vertices and vertex normals.
        vertices_within_chunk = vertices[map_from]
        vertex_normals_within_chunk = vertex_normals[map_from]

        # create mesh and triangles and add to scene
        mymesh = Mesh(material=_material)
        mesh_idx = mymesh.mesh_index

        triangle_count_start = _scene.count_triangles() 

        _decimation_factor = 0.9

        triangles = numba_scripts.classes.triangles_from_obj_data(
            facet_vertex_indices_within_chunk,
            vertices_within_chunk,
            vertex_normals_within_chunk,
            mesh_idx,
            triangle_count_start,
        )

        mymesh.csys = _csys
        mymesh.triangles = triangles

        _scene.meshes.append(mymesh)
        results.append(mymesh)

    return results


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numba_scripts.classes import ByteableObject, Triangle
    from ..classes import Mesh

from pyrr import Vector3

def bounding_box(tris:list[Triangle]) -> pyrr.aabb:
    pts = []
    for tri in tris:
        pts += tri.positions
    ret = pyrr.aabb.create_from_points(pts)
    return ret

# pyrr.aabb.create_from_points()

def point_in_aabb(point, aabb):
    return np.all(point >= aabb[0]) and np.all(point <= aabb[1])

def chunk_mesh_bvh(mesh:"Mesh"):
    g = BVHGraph(mesh=mesh)
    node0 = BVHParentNode(graph=g, tris=mesh.triangles, depth=0)
    node0.split_recursively(max_depth=4)
    mesh.bvh_graph = g
    pass

class BVHGraph(object):
    ID_COUNTER = 0
    BVH_TRI_ID_LIST_GLOBAL = []
    graphs:list[BVHGraph] = []
    
    def __init__(self, mesh:"Mesh"):
        BVHGraph.graphs.append(self)
        self.id = BVHGraph.ID_COUNTER
        BVHGraph.ID_COUNTER += 1
        self.mesh = mesh
        self.node_id = 0
        self.nodes:list[BVHParentNode] = []
        self.tri_id_list:list[int] = []
        self.leaf_nodes:list = None

    def add_tri_ids_get_start_offset(self, tri_ids) -> tuple[int, int]:
        start_offset = len(BVHGraph.BVH_TRI_ID_LIST_GLOBAL)
        BVHGraph.BVH_TRI_ID_LIST_GLOBAL += tri_ids
        return start_offset
    
    def register_leaf_nodes(self):
        self.leaf_nodes = []
        for node in self.nodes:
            if node.is_leaf():
                node.tris_start_offset = len(BVHGraph.BVH_TRI_ID_LIST_GLOBAL)
                BVHGraph.BVH_TRI_ID_LIST_GLOBAL += node.tri_ids.tolist()
                self.leaf_nodes.append(node)
                pass

    @staticmethod
    def register_all():
        for g in BVHGraph.graphs:
            g.register_leaf_nodes()

    @classmethod
    def reset(cls):
        cls.ID_COUNTER = 0
        cls.BVH_TRI_ID_LIST_GLOBAL = []


aabb_0 = pyrr.aabb.create_from_points(np.array([0.0, 0.0, 0.0]))
class BVHParentNode:

    def __init__(self, graph:BVHGraph, tris, depth):
        self.graph = graph
        self.node_id = self.graph.node_id
        self.graph.node_id += 1
        
        self.depth = depth

        self.aabb = None
        self.child_left:None|BVHParentNode = None
        self.child_right:None|BVHParentNode = None

        self.tris:list["Triangle"] = tris
        self.vertices:np.ndarray = None
        self.centroids:np.ndarray = None
        self.tri_ids:np.ndarray = None

        if not self.tris:
            pass

        self._update_tri_ids()
        self._update_vertices()
        self._update_centroids()
        self._update_aabb()

        self.graph.nodes.append(self)

        self.tris_count = len(tris)
        self.tris_start_offset = None

        # self.tris_count = len(self.tris)
        # self.tris_start_offset = self.graph.add_tri_ids_get_start_offset(self.tri_ids.tolist())

        pass

    def is_leaf(self):
        return self.child_left is None

    def _update_tri_ids(self):
        self.tri_ids = np.array([t.triangle_id for t in self.tris], dtype=np.int32)

    def split_recursively(self, max_depth):
        # if there are no triangles to split, don't split...
        if len(self.tris) <= 1:
            return
        if self.depth == max_depth:
            return
        self.split()
        if self.child_left is not None:
            self.child_left.split_recursively(max_depth=max_depth)
        if self.child_right is not None:
            self.child_right.split_recursively(max_depth=max_depth)
        pass

    def _update_vertices(self):
        self.vertices = np.array([x.positions for x in self.tris])

    def _update_centroids(self):
        if not self.tris:
            self.centroids = np.array([[]])
            return
        self.centroids = np.mean(self.vertices, axis=2)

    def _update_aabb(self):
        if not self.tris:
            self.aabb = aabb_0.copy()
            return
        self.aabb = pyrr.aabb.create_from_points(self.vertices.reshape(-1, 3))

    def split(self):
        vertices = self.vertices
        aabb = self.aabb

        if not vertices.size:
            return

        longest_axis = np.argmax(abs(aabb[0]-aabb[1]))

        diff = (aabb[1]-aabb[0]) * np.array([0 if i != longest_axis else 0.5 for i in range(3)], dtype=np.float32)

        aabb_left_init = np.array([aabb[0], aabb[1] - diff])

        tris_left = []
        tris_right = []
        for j, cent in enumerate(self.centroids):
            if point_in_aabb(cent, aabb_left_init):
                tris_left.append(j)
            else:
                tris_right.append(j)

        if tris_left:
            self.child_left = BVHParentNode(graph=self.graph, tris=[self.tris[x] for x in tris_left], depth=self.depth + 1)
        if tris_right:
            self.child_right = BVHParentNode(graph=self.graph, tris=[self.tris[x] for x in tris_right], depth=self.depth + 1)


    def tobytes(self):
        return struct.pack(
            "4f 3f i 4i",
            *self.aabb[0], 0.0,
            *self.aabb[1],
            self.node_id,
            getattr(self.child_left, "node_id", -1),
            getattr(self.child_right, "node_id", -1),
            (self.tris_start_offset if self.tris_start_offset is not None else -1),
            self.tris_count,
        )
    
