"""Download every AI model once into ./models so the app can ship with them.

    python fetch_models.py
"""
import os
from pathlib import Path

root = Path(__file__).parent / "models"
os.environ["REMBG_HOME"] = str(root / "rembg")
os.environ["PIXELCLIP_MODELS"] = str(root / "lab")

import lab
from rembg import new_session

for name in ("u2netp", "isnet-general-use", "birefnet-general"):
    print("Downloading", name, "...")
    new_session(name)
for key in lab.MODELS:
    print("Downloading", lab.MODELS[key]["label"], "...")
    lab.ensure(key)
print("Done. Models are in", root)
