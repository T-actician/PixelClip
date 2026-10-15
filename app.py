"""PixelClip - background remover. Run: python app.py"""
import io, ipaddress, socket, sys, threading, urllib.parse, urllib.request, webbrowser, zipfile
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps
from rembg import new_session, remove

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


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.post("/api/remove")
def remove_bg(file: UploadFile = File(...), model: str = Form("balanced"), matting: bool = Form(False)):
    try:
        img = ImageOps.exif_transpose(Image.open(file.file)).convert("RGBA")
    except Exception:
        raise HTTPException(400, "That file is not a readable image.")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    out = remove(
        img,
        session=session_for(model),
        alpha_matting=matting,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=15,
        alpha_matting_erode_size=8,
    )
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
