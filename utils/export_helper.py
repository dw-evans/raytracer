import shutil
from pathlib import Path
import re


src = Path() / "renders/RayTracerDynamic/XXXX"
dst = Path() / "renders/consumed"

last_frame = 1000
last_cycle = 1000


fps = list(src.glob("*.png")) + list(src.glob("*.exr"))

fps_filtered = []
for fp in fps:
    mtch = re.match(r"^f(\d{5})_c(\d{5})", fp.stem, re.IGNORECASE)
    if not mtch:
        raise Exception
    frame = int(mtch.group(1))
    cycle = int(mtch.group(2))

    # skip all with a new frame number
    if frame > last_frame:
        continue
    # skip the newer cycles if it is the same last frame
    if frame == last_frame:
        if cycle > last_cycle:
            continue
    
    dst_fp = dst / fp.name
    print(f"moving {fp} to {dst_fp}...")
    pass
    # shutil.move(fp, dst_fp)