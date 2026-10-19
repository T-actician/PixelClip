"""PixelClip - photo toolkit (cutouts, ID photos, resizing, Photo Lab). Run: python app.py"""
import io, ipaddress, socket, sys, threading, urllib.parse, urllib.request, webbrowser, zipfile
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
import numpy as np
from PIL import Image, ImageOps
from pymatting import estimate_foreground_ml
from rembg import new_session, remove
from scipy import ndimage as ndi

import lab  # Photo Lab: enhance, upscale, erase, restore, retouch, blur, privacy, vectorize

try:  # face detection for the passport/ID tool (optional: the rest works without it)
    import cv2
except Exception:
    cv2 = None

HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))  # works inside the .exe build too
MAX_BYTES = 25 * 1024 * 1024
MODELS = {
    "fast": "u2netp",                  # small, instant
    "balanced": "isnet-general-use",   # best speed/quality mix
    "ultra": "birefnet-general",       # sharpest edges, hair, fur (slower, bigger download)
}
_sessions, _lock = {}, threading.Lock()
_cascade = None
MAX_SIDE = 4096

app = FastAPI(title="PixelClip")


def session_for(key: str):
    if key not in MODELS:
        raise HTTPException(400, "Unknown model")
    with _lock:  # models download on first use, then stay cached in memory
        if key not in _sessions:
            _sessions[key] = new_session(MODELS[key])
        return _sessions[key]


def refine(img: Image.Image, mask: Image.Image, soft: bool = False) -> Image.Image:
    """Turn a rough mask into a clean, smooth cutout with no background halo."""
    a = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    a[a < 0.05] = 0.0                                    # kill faint haze around the subject
    solid = a > 0.5
    lab, n = ndi.label(solid)
    if n > 1:                                            # drop stray specks, keep real subjects
        sizes = ndi.sum(solid, lab, range(1, n + 1))
        keep = np.flatnonzero(sizes >= max(50, sizes.max() * 0.03)) + 1
        a *= ndi.binary_dilation(np.isin(lab, keep), iterations=4)
    lo, hi = (0.05, 0.90) if soft else (0.15, 0.85)
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)          # tighten the soft edge (removes halo)
    a = ndi.grey_erosion(a, size=(3, 3))                 # shave the last 1px of light fringe
    a = ndi.gaussian_filter(a, 1.0 if soft else 0.7)     # anti-alias so edges look smooth
    a[a < 0.02] = 0.0
    a[a > 0.98] = 1.0                                    # fully solid inside, no see-through haze
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    if img.width * img.height <= 12_000_000:             # swap background-colored edge pixels for true subject color
        rgb = estimate_foreground_ml(rgb, a)
    out = np.dstack([np.clip(rgb, 0, 1), a])
    out[a == 0, :3] = 0
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA")


def _subject_mask(rgb):
    """Subject mask (0-1, subject = 1) for the Photo Lab's background blur."""
    m = remove(Image.fromarray(rgb), session=session_for("balanced"), only_mask=True, post_process_mask=True)
    return np.asarray(m.convert("L"), dtype=np.float32) / 255.0


lab.bg_mask = _subject_mask
app.include_router(lab.router)


@app.get("/lab")
def lab_page():
    return FileResponse(HERE / "lab.html")


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.post("/api/remove")
def remove_bg(file: UploadFile = File(...), model: str = Form("balanced"), matting: bool = Form(False), clean: bool = Form(True)):
    try:
        img = ImageOps.exif_transpose(Image.open(file.file)).convert("RGBA")
    except Exception:
        raise HTTPException(400, "That file is not a readable image.")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    mask = remove(img, session=session_for(model), only_mask=True, post_process_mask=True)
    if clean:
        out = refine(img, mask, soft=matting)
    else:
        out = img.copy()
        out.putalpha(mask)
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.post("/api/face")
def find_face(file: UploadFile = File(...)):
    """Find the largest front-facing face. Returns a box as fractions (0-1) of the image."""
    if cv2 is None:
        return {"found": False}
    try:
        img = ImageOps.exif_transpose(Image.open(file.file)).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file is not a readable image.")
    scale = min(1.0, 1000 / max(img.size))
    if scale < 1:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    gray = cv2.equalizeHist(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY))
    global _cascade
    with _lock:
        if _cascade is None:
            _cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    h, w = gray.shape
    side = max(30, int(w * 0.08))
    faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(side, side))
    if len(faces) == 0:
        return {"found": False}
    x, y, fw, fh = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    return {"found": True, "x": float(x) / w, "y": float(y) / h, "w": float(fw) / w, "h": float(fh) / h}


@app.get("/api/fetch")
def fetch_url(url: str):
    """Download an image from a link (done server-side to avoid browser CORS limits)."""
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise HTTPException(400, "Enter a full link starting with http:// or https://")
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(u.hostname))
    except Exception:
        raise HTTPException(400, "Could not reach that address.")
    if ip.is_private or ip.is_loopback or ip.is_link_local:
        raise HTTPException(400, "Links to local network addresses are blocked.")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PixelClip"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data, ctype = r.read(MAX_BYTES + 1), r.headers.get_content_type()
    except Exception:
        raise HTTPException(400, "Could not download that link.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "That image is larger than 25 MB.")
    if not ctype.startswith("image/"):
        raise HTTPException(400, "That link is not an image.")
    return Response(data, media_type=ctype)


@app.post("/api/zip")
def make_zip(files: List[UploadFile] = File(...)):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.writestr(f.filename, f.file.read())
    return Response(buf.getvalue(), media_type="application/zip")


if __name__ == "__main__":
    port = 8765
    threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
