"""Build a standalone PixelClip app (no Python needed to run it).

    pip install pyinstaller
    python build.py

Output: dist/PixelClip/PixelClip.exe (Windows) or dist/PixelClip/PixelClip (macOS/Linux).
Share the whole dist/PixelClip folder.
"""
import os
import PyInstaller.__main__

PyInstaller.__main__.run([
    "app.py",
    "--name", "PixelClip",
    "--noconfirm",
    "--onedir",
    "--add-data", f"index.html{os.pathsep}.",
    "--add-data", f"lab.html{os.pathsep}.",
    "--add-data", f"models{os.pathsep}models",
    "--collect-all", "rembg",
    "--collect-all", "pymatting",
    "--collect-all", "onnxruntime",
    "--collect-all", "numba",
    "--collect-all", "llvmlite",
    "--collect-all", "cv2",
    "--collect-all", "vtracer",
])
