from scripts.classes import Csys
from pyrr import Vector3
import pyrr

x = Csys()

x.Rx(30)
x.Ry(10)
x.Rz(15)
x.translate(Vector3((10, 20, 30)))

pass
