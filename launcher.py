"""PixelClip launcher.

The screen opens in about a second. The heavy parts (image engine, AI models) load in the
background; any request that needs them simply waits until they are ready.
"""
import asyncio, os, socket, sys, threading, time, traceback, webbrowser
from pathlib import Path

PORT = 8765
URL = f"http://127.0.0.1:{PORT}"
HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
DATA = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "PixelClip"
try:
    DATA.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(DATA / "numba"))  # compile once, reuse on later launches
except Exception:
    pass

core = {"app": None, "err": ""}
seen = {"t": None}


def log_error(text):
    core["err"] = text
    try:
        (DATA / "error.log").write_text(text, encoding="utf-8")
    except Exception:
        pass


def load_core():
    """Runs in the background: import the real app, then warm it up."""
    try:
        from app import app as real  # the heavy part
        core["app"] = real
    except BaseException:
        log_error(traceback.format_exc())
        return
    try:  # warm-up so the first click is fast
        import numpy as np
        from pymatting import estimate_foreground_ml
        estimate_foreground_ml(np.full((24, 24, 3), .5, np.float32), np.full((24, 24), .5, np.float32))
    except Exception:
        pass
    try:
        sys.modules["app"].session_for("balanced")
    except Exception:
        pass


async def shell(scope, receive, send):
    from starlette.responses import FileResponse, JSONResponse
    kind = scope["type"]
    if kind == "lifespan":
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
                threading.Thread(target=load_core, daemon=True).start()
                threading.Thread(target=lambda: webbrowser.open(URL), daemon=True).start()
            elif msg["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    if kind != "http":
        return
    path = scope.get("path", "")
    if path in ("/", "/index.html"):
        return await FileResponse(HERE / "index.html", headers={"Cache-Control": "no-store"})(scope, receive, send)
    if path == "/lab":
        return await FileResponse(HERE / "lab.html", headers={"Cache-Control": "no-store"})(scope, receive, send)
    if path == "/api/ping":
        seen["t"] = time.time()
        return await JSONResponse({"ok": True})(scope, receive, send)
    if path == "/api/ready":
        return await JSONResponse({"ready": core["app"] is not None, "error": core["err"][-300:]})(scope, receive, send)
    waited = 0.0
    while core["app"] is None and not core["err"] and waited < 300:
        await asyncio.sleep(0.1)
        waited += 0.1
    if core["app"] is None:
        msg = "PixelClip could not start: " + (core["err"][-300:] or "it took too long to load.")
        return await JSONResponse({"detail": msg}, status_code=500)(scope, receive, send)
    await core["app"](scope, receive, send)


def watchdog():  # quit when the PixelClip browser tab has been closed for 3 minutes
    while True:
        time.sleep(15)
        if seen["t"] and time.time() - seen["t"] > 180:
            os._exit(0)


if __name__ == "__main__":
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:  # already running: just open it
            webbrowser.open(URL)
            sys.exit(0)
    if sys.stdout is None:  # no console window: send output nowhere
        sys.stdout = sys.stderr = open(os.devnull, "w")
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        import uvicorn
        uvicorn.run(shell, host="127.0.0.1", port=PORT, log_level="warning", log_config=None, lifespan="on")
    except BaseException:
        log_error(traceback.format_exc())
