
from __future__ import annotations
import trimesh
from ..classes import Material, Scene, Mesh
import pyrr
import numpy as np
import numba_scripts.classes

from numba_scripts.classes import Csys


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

