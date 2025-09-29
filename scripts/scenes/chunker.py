
from __future__ import annotations
import trimesh
import pyrr
import numpy as np
import struct

from scripts.classes import Material, Scene, Mesh
import scripts.numba_utils.classes
from scripts.numba_utils.classes import Csys
from scripts.numba_utils.functions import timer


def point_in_aabb(point, aabb):
    return np.all(point >= aabb[0]) and np.all(point <= aabb[1])

def chunk_mesh_bvh(mesh:"Mesh"):
    g = BVHGraph(mesh=mesh)
    if not mesh.triangles:
        print("Warning, no triangles detected in mesh, returning without mesh!")
        raise Exception
        return
    node0 = BVHParentNode(graph=g, tris=mesh.triangles, depth=0)
    node0.split_recursively(max_depth=20)
    mesh.bvh_graph = g
    # plot_aabbs([x.aabb for x in g.nodes])
    pass

class BVHGraph(object):
    BVH_TRI_ID_LIST_GLOBAL = []
    ALL:list[BVHGraph] = []
    
    def __init__(self, mesh:"Mesh"):
        self.id = len(BVHGraph.ALL)
        BVHGraph.ALL.append(self)
        
        self.mesh = mesh
        self._nodes:list[BVHParentNode] = []
        # self.tri_id_list:list[int] = []
        self.leaf_nodes:list[BVHParentNode] = None
        # self.node_start_index_map = {}
        self.is_awaiting_mesh_update = False

    def flag_for_mesh_update(self):
        self.is_awaiting_mesh_update = True

    def unflag_for_mesh_update(self):
        self.is_awaiting_mesh_update = False

    @property
    def first_node_id(self):
        return self._nodes[0].id
    
    @classmethod
    def reset(cls):
        cls.ALL = []
        cls.BVH_TRI_ID_LIST_GLOBAL = []

    def register_node(self, obj):
        self._nodes.append(obj)

    def add_tri_ids_get_start_offset(self, tri_ids) -> tuple[int, int]:
        raise NotImplementedError
        start_offset = len(BVHGraph.BVH_TRI_ID_LIST_GLOBAL)
        BVHGraph.BVH_TRI_ID_LIST_GLOBAL += tri_ids
        return start_offset
    
    def update_leaf_nodes_and_tri_ids(self):
        self.leaf_nodes = []
        for node in self._nodes:
            if node.is_leaf():
                node._update_tri_ids()
                self.leaf_nodes.append(node)
                pass
        pass



    @staticmethod
    def register_all_and_update_node_tri_ids():
        BVHGraph.BVH_TRI_ID_LIST_GLOBAL = []
        for g in BVHGraph.ALL:
            g.update_leaf_nodes_and_tri_ids()
            for node in g.leaf_nodes:
                node.tris_start_offset = len(BVHGraph.BVH_TRI_ID_LIST_GLOBAL)
                BVHGraph.BVH_TRI_ID_LIST_GLOBAL += node.tri_ids.tolist()


    @staticmethod
    def reset():
        BVHGraph.BVH_TRI_ID_LIST_GLOBAL = []
        BVHGraph.ALL = [] 

    def update_graph_node_aabbs_for_changed_triangles(self, force=False):
        # print("updating node graph for modified triangles")
        def fn():
            if (not force) and (not self.is_awaiting_mesh_update):
                return
            
            for i, x in enumerate(self._nodes):
                x.update_for_modified_triangles()
                # print(i)

            self.unflag_for_mesh_update()

        fn()
        # timer(fn)()
        # print(f"updated {len(BVHGraph.ALL)} node aabbs")



AABB_NULL = pyrr.aabb.create_from_points(np.array([0.0, 0.0, 0.0]))
class BVHParentNode:
    ALL:list[BVHParentNode] = []

    def __init__(self, graph:BVHGraph, tris, depth):
        self.graph = graph
        self.id = len(BVHParentNode.ALL)
        BVHParentNode.ALL.append(self)

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

        self.graph.register_node(self)

        self.tris_count = len(tris)
        self.tris_start_offset = None

        # self.tris_count = len(self.tris)
        # self.tris_start_offset = self.graph.add_tri_ids_get_start_offset(self.tri_ids.tolist())
        pass

    @classmethod
    def reset(cls):
        cls.ALL = []

    @property
    def child_left_id(self):
        return getattr(self.child_left, "id", -1)
    @property
    def child_right_id(self):
        return getattr(self.child_right, "id", -1)

    def is_leaf(self):
        return self.child_left is None and self.child_right is None

    def _update_tri_ids(self):
        self.tri_ids = np.array([t.id for t in self.tris], dtype=np.int32)

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

    @staticmethod
    def update_aabs_for_changed_triangles(force=False):
        print("updating node graph for modified triangles")
        def fn():
            for i, x in enumerate(BVHParentNode.ALL):
                if force or x.graph.is_awaiting_mesh_update:
                    x.update_for_modified_triangles()
                else:
                    pass
        timer(fn)()
        print(f"updated {len(BVHGraph.ALL)} node aabbs")

    def update_for_modified_triangles(self):
        self._update_vertices()
        self._update_aabb()

    def _update_vertices(self):
        self.vertices = np.array([x.positions for x in self.tris])

    def _update_centroids(self):
        if not self.tris:
            self.centroids = np.array([[]])
            return
        self.centroids = np.mean(self.vertices, axis=1)

    def _update_aabb(self):
        if not self.tris:
            self.aabb = AABB_NULL.copy()
            return
        self.aabb = pyrr.aabb.create_from_points(self.vertices.reshape(-1, 3))

    def split(self):
        aabb = self.aabb

        longest_axis = np.argmax(abs(aabb[0]-aabb[1]))

        # median_centroid_position = np.median(self.centroids[:,longest_axis])

        diff = (aabb[1]-aabb[0]) * np.array([0 if i != longest_axis else 0.5 for i in range(3)], dtype=np.float32)


        aabb_left_init = np.array([aabb[0], aabb[1] - diff])

        tris_left = []
        tris_right = []
        for j, cent in enumerate(self.centroids):
            if point_in_aabb(cent, aabb_left_init):
                tris_left.append(j)
            else:
                tris_right.append(j)

        # the length of a child is equal to the parent. No splitting has occurred!
        if (not tris_left) or (not tris_right):
            # return without making the children
            return

        if tris_left:
            self.child_left = BVHParentNode(graph=self.graph, tris=[self.tris[x] for x in tris_left], depth=self.depth + 1)
        if tris_right:
            self.child_right = BVHParentNode(graph=self.graph, tris=[self.tris[x] for x in tris_right], depth=self.depth + 1)


    def tobytes(self):
        return struct.pack(
            "4f 3f i 4i",
            *self.aabb[0], 0.0,
            *self.aabb[1],
            self.id,
            self.child_left_id,
            self.child_right_id,
            (self.tris_start_offset if self.tris_start_offset is not None else -1),
            self.tris_count,
        )
    

    
