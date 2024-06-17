from pathlib import Path
from stl import mesh
import numpy as np

file = Path() / "objects/warped_cube.stl"

mesh_data = mesh.Mesh.from_file(file)

from pyrr import Vector3

triangles = []
for facet in mesh_data.data:
    facet: np.ndarray
    normal = Vector3(facet[0])
    normal.normalize()
    v0, v1, v2 = [Vector3(x) for x in facet[1]]
    pass

if __name__ == "__main__":
    pass

pass
