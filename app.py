"""Cutline - background remover. Run: python app.py"""
import io, threading, webbrowser, zipfile
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps
from rembg import new_session, remove

HERE = Path(__file__).parent
MODELS = {
    "fast": "u2netp",                  # small, instant
    "balanced": "isnet-general-use",   # best speed/quality mix
    "ultra": "birefnet-general",       # sharpest edges, hair, fur (slower, bigger download)
}
_sessions, _lock = {}, threading.Lock()
MAX_SIDE = 4096

app = FastAPI(title="Cutline")


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
