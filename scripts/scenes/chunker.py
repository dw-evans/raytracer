
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
    plot_aabbs([x.aabb for x in g.nodes])
    pass

import matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import cycler
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
my_cycler = cycler.cycler(color=colors)
import itertools
# Apply it globally


def plot_aabbs(aabbs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    lines = []
    for bbmin, bbmax in aabbs:
        # Get corners of the box
        x0, y0, z0 = bbmin
        x1, y1, z1 = bbmax
        corners = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ])

        # Define edges as pairs of corner indices
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]

        for e in edges:
            lines.append([corners[e[0]], corners[e[1]]])

    # Add all edges at once
    c = itertools.cycle(my_cycler)

    # lc = Line3DCollection(lines, colors=c, linewidths=1)
    lc = Line3DCollection(lines, colors="blue", linewidths=1)
    ax.add_collection3d(lc)

    # Auto scale axes
    all_points = np.array([p for line in lines for p in line])
    ax.auto_scale_xyz(all_points[:,0], all_points[:,1], all_points[:,2])

    # plt.ion()  # interactive mode ON
    plt.show()
    pass


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def plot_aabbs_filled(aabbs, face_color="cyan", edge_color="k", alpha=0.3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for bbmin, bbmax in aabbs:
        x0, y0, z0 = bbmin
        x1, y1, z1 = bbmax
        corners = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ])

        # Define the 6 cube faces
        faces = [
            [corners[j] for j in [0,1,2,3]],  # bottom
            [corners[j] for j in [4,5,6,7]],  # top
            [corners[j] for j in [0,1,5,4]],  # front
            [corners[j] for j in [2,3,7,6]],  # back
            [corners[j] for j in [1,2,6,5]],  # right
            [corners[j] for j in [0,3,7,4]],  # left
        ]

        ax.add_collection3d(
            Poly3DCollection(faces, 
                             facecolors=face_color, 
                             edgecolors=edge_color, 
                             linewidths=1, 
                             alpha=alpha)
        )

    # Auto scale axes to fit all boxes
    all_points = np.array([c for bb in aabbs for c in [bb[0], bb[1]]])
    ax.auto_scale_xyz(all_points[:,0], all_points[:,1], all_points[:,2])

    plt.show()






class BVHGraph(object):
    ID_COUNTER = 0
    BVH_TRI_ID_LIST_GLOBAL = []
    GRAPHS:list[BVHGraph] = []
    
    def __init__(self, mesh:"Mesh"):
        BVHGraph.GRAPHS.append(self)
        self.id = BVHGraph.ID_COUNTER
        BVHGraph.ID_COUNTER += 1
        self.mesh = mesh
        self.node_id_counter = 0
        self.nodes:list[BVHParentNode] = []
        self.tri_id_list:list[int] = []
        self.leaf_nodes:list = None

    def add_tri_ids_get_start_offset(self, tri_ids) -> tuple[int, int]:
        start_offset = len(BVHGraph.BVH_TRI_ID_LIST_GLOBAL)
        BVHGraph.BVH_TRI_ID_LIST_GLOBAL += tri_ids
        return start_offset
    
    def update_leaf_tris_and_register(self):
        self.leaf_nodes = []
        for node in self.nodes:
            if node.is_leaf():
                node._update_tri_ids()
                node.tris_start_offset = len(BVHGraph.BVH_TRI_ID_LIST_GLOBAL)
                BVHGraph.BVH_TRI_ID_LIST_GLOBAL += node.tri_ids.tolist()
                self.leaf_nodes.append(node)
                pass
        pass

    @staticmethod
    def register_all():
        for g in BVHGraph.GRAPHS:
            g.update_leaf_tris_and_register()

    @classmethod
    def reset(cls):
        cls.ID_COUNTER = 0
        cls.BVH_TRI_ID_LIST_GLOBAL = []
        cls.GRAPHS = []


aabb_0 = pyrr.aabb.create_from_points(np.array([0.0, 0.0, 0.0]))
class BVHParentNode:

    def __init__(self, graph:BVHGraph, tris, depth):
        self.graph = graph
        self.node_id = self.graph.node_id_counter
        self.graph.node_id_counter += 1
        
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
        return self.child_left is None and self.child_right is None

    def _update_tri_ids(self):
        self.tri_ids = np.array([t.triangle_id for t in self.tris], dtype=np.int32)

    def split_recursively(self, max_depth):
        # if there are no triangles to split, don't split...
        if len(self.tris) <= 5:
            return
        if self.depth == max_depth:
            return
        print(f"self.tris.__len__(): {self.tris.__len__()}")
        self.split()
        if self.child_left is not None:
            print(f"self.child_left.tris.__len__(): {self.child_left.tris.__len__()}")
        if self.child_right is not None:
            print(f"self.child_right.tris.__len__(): {self.child_right.tris.__len__()}")

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
        self.centroids = np.mean(self.vertices, axis=1)

    def _update_aabb(self):
        if not self.tris:
            self.aabb = aabb_0.copy()
            return
        self.aabb = pyrr.aabb.create_from_points(self.vertices.reshape(-1, 3))

    def split(self):
        vertices = self.vertices
        aabb = self.aabb

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

        # tl = [self.tris[x] for x in tris_left]
        # tr = [self.tris[x] for x in tris_right]

        # vl = np.array([x.positions for x in tl])
        # vr = np.array([x.positions for x in tr])

        # aabbl = pyrr.aabb.create_from_points(vl.reshape(-1, 3))
        # aabbr = pyrr.aabb.create_from_points(vr.reshape(-1, 3))

        if tris_left:
            self.child_left = BVHParentNode(graph=self.graph, tris=[self.tris[x] for x in tris_left], depth=self.depth + 1)
        if tris_right:
            self.child_right = BVHParentNode(graph=self.graph, tris=[self.tris[x] for x in tris_right], depth=self.depth + 1)

        pass


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
    
