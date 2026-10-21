"""PixelClip launcher: opens the browser instantly with a loading page while the app loads."""
import os, sys, threading, time, traceback, webbrowser
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8765
ERR = {"text": ""}
DATA = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "PixelClip"
try:
    DATA.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(DATA / "numba"))  # compile once, reuse next launches
except Exception:
    pass
URL = f"http://127.0.0.1:{PORT}"
PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>PixelClip</title>
<style>body{margin:0;height:100vh;display:grid;place-items:center;background:#0f1b22;color:#fff;font:16px system-ui,sans-serif;text-align:center}
.s{width:42px;height:42px;margin:0 auto 18px;border:4px solid #ff4d8d;border-right-color:transparent;border-radius:50%;animation:r .8s linear infinite}
@keyframes r{to{transform:rotate(360deg)}}small{opacity:.6}</style></head>
<body><div><div class="s"></div><b>Starting PixelClip...</b><br><small>This page opens by itself. The first launch can take a little longer.</small></div>
<pre id="e" style="display:none;max-width:90vw;margin:18px auto 0;text-align:left;white-space:pre-wrap;color:#ff9db8;font-size:12px"></pre><!--__loading__--><script>setInterval(function(){fetch("/__err").then(function(r){return r.text()}).then(function(t){if(t){var e=document.getElementById("e");e.style.display="block";e.textContent="PixelClip could not start:\\n"+t}}).catch(function(){});fetch("/?p="+Date.now()).then(function(r){return r.text()}).then(function(t){if(t.indexOf("__loading__")<0)location.reload()}).catch(function(){})},700)</script></body></html>"""


class Loading(BaseHTTPRequestHandler):
    def do_GET(self):
        body, kind = (ERR["text"].encode("utf-8", "replace"), "text/plain") if self.path.startswith("/__err") else (PAGE, "text/html")
        self.send_response(200)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", PORT), Loading)
    except OSError:  # PixelClip is already running: just open it
        webbrowser.open(URL)
        sys.exit(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    webbrowser.open(URL)
    try:
        if sys.stdout is None:  # no console window: send output nowhere
            sys.stdout = sys.stderr = open(os.devnull, "w")
        import uvicorn
        from app import app as pixelclip  # the heavy part
        seen = {"t": None}

        @pixelclip.get("/api/ping")
        def ping():
            seen["t"] = time.time()
            return {"ok": True}

        def watchdog():  # quit when the PixelClip browser tab has been closed for 3 minutes
            while True:
                time.sleep(15)
                if seen["t"] and time.time() - seen["t"] > 180:
                    os._exit(0)

        threading.Thread(target=watchdog, daemon=True).start()
        srv.shutdown()
        srv.server_close()
        uvicorn.run(pixelclip, host="127.0.0.1", port=PORT, log_level="warning", log_config=None)
    except BaseException:
        ERR["text"] = traceback.format_exc()
        try:
            (DATA / "error.log").write_text(ERR["text"], encoding="utf-8")
        except Exception:
            pass
        threading.Event().wait()  # keep the page up so it can show the error
