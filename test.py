from scripts.classes import Csys
from pyrr import Vector3
import pyrr

x = Csys()

x.Rx(30)
x.Ry(10)
x.Rz(15)
x.translate(Vector3((10, 20, 30)))

pass

# s = ""
# s2 = """
# """
# for i in range(15):
#     s += f"layout(std140) uniform triBuffer{i}\n{{\n    Triangle triangles{i}[TRIANGLES_COUNT_MAX]\n}};\n"
#     s2 += f"""
#     else if (index < TRIANGLES_COUNT_MAX * {i + 1})
#     {{
#         ret = triangles{i}[index - TRIANGLES_COUNT_MAX * {i}];
#     }}"""


# print(s)
# print(s2)


pass
