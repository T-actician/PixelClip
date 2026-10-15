"""PixelClip - background remover. Run: python app.py"""
import io, ipaddress, socket, sys, threading, urllib.parse, urllib.request, webbrowser, zipfile
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
import numpy as np
from PIL import Image, ImageOps
from pymatting import estimate_alpha_cf, estimate_foreground_ml
from rembg import new_session, remove
from scipy import ndimage as ndi

HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))  # works inside the .exe build too
MAX_BYTES = 25 * 1024 * 1024
MODELS = {
    "fast": "u2netp",                  # small, instant
    "balanced": "isnet-general-use",   # best speed/quality mix
    "ultra": "birefnet-general",       # sharpest edges, hair, fur (slower, bigger download)
}
_sessions, _lock = {}, threading.Lock()
MAX_SIDE = 4096

app = FastAPI(title="PixelClip")


def session_for(key: str):
    if key not in MODELS:
        raise HTTPException(400, "Unknown model")
    with _lock:  # models download on first use, then stay cached in memory
        if key not in _sessions:
            _sessions[key] = new_session(MODELS[key])
        return _sessions[key]


def drop_specks(solid: np.ndarray) -> np.ndarray:
    """Keep real subjects, drop tiny stray blobs."""
    lab, n = ndi.label(solid)
    if n <= 1:
        return solid
    sizes = ndi.sum(solid, lab, range(1, n + 1))
    keep = np.flatnonzero(sizes >= max(50, sizes.max() * 0.03)) + 1
    return np.isin(lab, keep)


def matte(img: Image.Image, solid: np.ndarray, wide: bool) -> np.ndarray:
    """Re-measure the true edge from the photo's own colors inside a thin band around the mask."""
    h, w = solid.shape
    f = min(1.0, (3_000_000 / (h * w)) ** 0.5)           # keep matting fast on big photos
    sw, sh = max(1, int(w * f)), max(1, int(h * f))
    small = img.convert("RGB").resize((sw, sh), Image.LANCZOS) if f < 1 else img.convert("RGB")
    rgb = np.asarray(small, dtype=np.float64) / 255.0
    sol = np.asarray(Image.fromarray((solid * 255).astype(np.uint8)).resize((sw, sh), Image.LANCZOS)) > 127 if f < 1 else solid
    r = max(4, int(round(0.008 * max(sw, sh) * (2.2 if wide else 1.0))))
    trimap = np.full(sol.shape, 0.5)
    trimap[ndi.binary_erosion(sol, iterations=r)] = 1.0   # surely subject
    trimap[~ndi.binary_dilation(sol, iterations=r)] = 0.0  # surely background
    a = np.clip(estimate_alpha_cf(rgb, trimap), 0, 1).astype(np.float32)
    if f < 1:
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0
    return a


def refine(img: Image.Image, mask: Image.Image, wide: bool = False) -> Image.Image:
    """Turn a rough mask into a clean, smooth cutout with no background halo."""
    m = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    m[m < 0.05] = 0.0                                    # kill faint haze
    solid = drop_specks(m > 0.5)
    try:
        a = matte(img, solid, wide)                      # edge measured from the real photo
        a *= ndi.binary_dilation(solid, iterations=int(0.012 * max(m.shape)) + 6)  # nothing far from the subject
    except Exception:                                    # fallback: simple mask tightening
        a = np.clip((m * ndi.binary_dilation(solid, iterations=4) - 0.15) / 0.70, 0.0, 1.0)
        a = ndi.gaussian_filter(a, 0.6)
    a[a < 0.03] = 0.0
    a[a > 0.97] = 1.0                                    # fully solid inside, no see-through haze
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    if img.width * img.height <= 12_000_000:             # replace background-colored edge pixels with true subject color
        rgb = estimate_foreground_ml(rgb, a)
    out = np.dstack([np.clip(rgb, 0, 1), a])
    out[a == 0, :3] = 0
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA")


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
        out = refine(img, mask, wide=matting)
    else:
        out = img.copy()
        out.putalpha(mask)
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


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
