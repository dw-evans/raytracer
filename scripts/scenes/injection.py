

from math import sin, cos, degrees, radians, pi
import numpy as np

from ..classes import (
    Camera,
)

def main(obj:Camera, row):
    obj.csys.set_pos([row["Location Y"], row["Location Z"], -row["Location X"]])
    obj.csys.quat = np.array([0, 0, 0, 1], dtype=np.float32)
    obj.csys.rxp(degrees(row["Rotation Y"]))
    obj.csys.ryp(degrees(row["Rotation Z"]-pi/2))
    obj.csys.rxp(degrees(row["Rotation X"]-pi/2))
    pass

