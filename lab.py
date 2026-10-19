"""PixelClip Photo Lab - enhance, upscale, erase, restore, retouch, blur, privacy, vectorize.

Everything runs on your machine. AI models download once into ~/.pixelclip/models.
"""
import gc, hashlib, io, json, os, threading, time, traceback, urllib.request, uuid, zlib
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None
try:
    import vtracer
except Exception:  # pragma: no cover
    vtracer = None

router = APIRouter(prefix="/api/lab")

MAX_BYTES = 40 * 1024 * 1024
MAX_SIDE = 8000
MODEL_DIR = Path(os.environ.get("PIXELCLIP_MODELS") or Path.home() / ".pixelclip" / "models")
FF = "https://github.com/facefusion/facefusion-assets/releases/download/"
MODELS = {
    "upscale": {"label": "upscaler", "file": "span_kendata_x4.onnx", "mb": 2, "crc": "a0d53205",
                "urls": [FF + "models-3.0.0/span_kendata_x4.onnx"]},
    "face": {"label": "face restorer", "file": "gfpgan_1.4.onnx", "mb": 340, "crc": "5a6c6364",
             "urls": [FF + "models-3.0.0/gfpgan_1.4.onnx"]},
    "detect": {"label": "face finder", "file": "yunet_2023_mar.onnx", "mb": 1, "crc": "8de24a30",
               "urls": [FF + "models-3.4.0/yunet_2023_mar.onnx"]},
    "lama": {"label": "object remover", "file": "lama_fp32.onnx", "mb": 208,
             "sha256": "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
             "urls": ["https://github.com/neureps/carousel-fit-models/releases/download/lama-v1/lama-fp32-1faef5301d78.onnx",
                      "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"]},
    "color": {"label": "colorizer", "file": "ddcolor.onnx", "mb": 980, "crc": "cab3b659",
              "urls": [FF + "models-3.0.0/ddcolor.onnx"]},
}


class LabError(Exception):
    """A problem the user can understand (shown as-is)."""


# ---------------------------------------------------------------- jobs
class Job:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.state, self.pct, self.msg, self.error, self.note = "queued", 0, "Waiting...", "", ""
        self.data, self.mime, self.t = None, "image/png", time.time()

    def upd(self, pct=None, msg=None):
        if pct is not None:
            self.pct = int(max(0, min(100, pct)))
        if msg is not None:
            self.msg = msg

    def add_note(self, text):
        self.note = (self.note + " " + text).strip()


_jobs, _jobs_lock, _run_lock = {}, threading.Lock(), threading.Lock()


def _prune():
    now = time.time()
    with _jobs_lock:
        for k in [k for k, j in _jobs.items() if now - j.t > 1800 and j.state in ("done", "error")]:
            del _jobs[k]
        while len(_jobs) > 40:
            del _jobs[min(_jobs, key=lambda k: _jobs[k].t)]


def _worker(job, fn, img, mask, params):
    with _run_lock:  # one heavy job at a time keeps the machine responsive
        job.state, job.msg = "running", "Starting..."
        try:
            job.data, job.mime = fn(job, img, mask, params)
            job.state, job.pct, job.msg = "done", 100, "Done"
        except LabError as e:
            job.state, job.error = "error", str(e)
        except Exception as e:
            traceback.print_exc()
            job.state, job.error = "error", "Something went wrong while processing this image (" + type(e).__name__ + ")."
        finally:
            with _slock:  # hand big models' memory back as soon as the job ends (reloading takes a few seconds)
                for k in [k for k in _sessions if k in BIG]:
                    del _sessions[k]
            gc.collect()


# ---------------------------------------------------------------- models
_dl_locks = {k: threading.Lock() for k in MODELS}
BIG = {"face", "lama", "color"}
_sessions, _slock = {}, threading.Lock()


def model_path(key):
    return MODEL_DIR / MODELS[key]["file"]


def _verify(path, m):
    if "crc" in m:
        c = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                c = zlib.crc32(chunk, c)
        return "%08x" % (c & 0xFFFFFFFF) == m["crc"]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == m["sha256"]


def ensure(key, job=None):
    """Make sure a model file is on disk, downloading it (with progress) the first time."""
    m, path = MODELS[key], model_path(key)
    with _dl_locks[key]:
        if path.exists():
            return path
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        part, last = path.with_suffix(".part"), "no address worked"
        for url in m["urls"]:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "PixelClip"})
                with urllib.request.urlopen(req, timeout=30) as r, open(part, "wb") as out:
                    total = int(r.headers.get("Content-Length") or 0) or m["mb"] * 1048576
                    got = 0
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        got += len(chunk)
                        if job:
                            job.upd(got * 100 / total, f"Downloading the {m['label']} ({m['mb']} MB, one time only)... {min(99, int(got * 100 / total))}%")
                if _verify(part, m):
                    part.replace(path)
                    return path
                last = "the download was damaged"
            except Exception as e:
                last = str(e)
            part.unlink(missing_ok=True)
        raise LabError(f"Could not download the {m['label']} ({last}). Check your internet connection and try again.")


def session(key, job=None):
    if ort is None:
        raise LabError("onnxruntime is not installed.")
    path = ensure(key, job)
    with _slock:
        if key not in _sessions:
            if key in BIG:  # keep memory modest: only one large model stays loaded at a time
                for k in [k for k in _sessions if k in BIG]:
                    del _sessions[k]
            prov = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in ort.get_available_providers()]
            _sessions[key] = ort.InferenceSession(str(path), providers=prov + ["CPUExecutionProvider"])
        return _sessions[key]


def models_status():
    return {k: {"label": m["label"], "mb": m["mb"], "ready": model_path(k).exists()} for k, m in MODELS.items()}


# ---------------------------------------------------------------- image helpers
def load_rgb(data, maxside=MAX_SIDE):
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            bg.alpha_composite(im)
            im = bg
        im = im.convert("RGB")
    except Exception:
        raise LabError("That file is not a readable image.")
    if max(im.size) > maxside:
        im.thumbnail((maxside, maxside), Image.LANCZOS)
    return np.array(im)


def load_mask(data, shape):
    try:
        m = np.array(Image.open(io.BytesIO(data)).convert("L"))
    except Exception:
        raise LabError("The selection could not be read.")
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m


def to_png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG", compress_level=3)
    return buf.getvalue()


def smooth(x, a, b):
    t = np.clip((x - a) / max(b - a, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def tiled(fn, img, tile=1024, pad=24):
    """Run a same-size filter on big images in overlapping pieces to keep memory and time bounded."""
    h, w = img.shape[:2]
    if h * w <= tile * tile * 1.5:
        return fn(img)
    out = np.empty_like(img)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y0, x0, y1, x1 = max(0, y - pad), max(0, x - pad), min(h, y + tile + pad), min(w, x + tile + pad)
            r = fn(img[y0:y1, x0:x1])
            out[y:min(h, y + tile), x:min(w, x + tile)] = r[y - y0:y - y0 + min(tile, h - y), x - x0:x - x0 + min(tile, w - x)]
    return out


# ---------------------------------------------------------------- faces
FFHQ = np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                 [0.39308822, 0.72541100], [0.61150205, 0.72490465]], np.float32) * 512


def _detect_once(sess, rgb, min_score):
    h, w = rgb.shape[:2]
    sc = 640 / max(h, w)
    r = cv2.resize(rgb, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
    can = np.zeros((640, 640, 3), np.float32)
    can[:r.shape[0], :r.shape[1]] = cv2.cvtColor(r, cv2.COLOR_RGB2BGR)
    o = sess.run(None, {"input": can.transpose(2, 0, 1)[None]})
    boxes, scores, kps = [], [], []
    for i, st in enumerate((8, 16, 32)):
        cls, obj, bb, kp = o[i][0, :, 0], o[i + 3][0, :, 0], o[i + 6][0], o[i + 9][0]
        sco = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
        n = 640 // st
        idx = np.arange(len(sco))
        col, row = idx % n, idx // n
        cx, cy = (col + bb[:, 0]) * st, (row + bb[:, 1]) * st
        bw, bh = np.exp(bb[:, 2]) * st, np.exp(bb[:, 3]) * st
        for j in np.nonzero(sco > min_score)[0]:
            boxes.append([float(cx[j] - bw[j] / 2) / sc, float(cy[j] - bh[j] / 2) / sc, float(bw[j]) / sc, float(bh[j]) / sc])
            scores.append(float(sco[j]))
            kps.append(np.stack([(kp[j, 0::2] + col[j]) * st, (kp[j, 1::2] + row[j]) * st], 1) / sc)
    return boxes, scores, kps


def detect_faces(rgb, job=None, min_score=0.7):
    """YuNet face finder run directly through onnxruntime. Returns boxes plus 5 landmarks per face.
    Small faces in big photos are found by also scanning the photo in native-resolution tiles."""
    sess = session("detect", job)
    h, w = rgb.shape[:2]
    boxes, scores, kps = _detect_once(sess, rgb, min_score)
    if max(h, w) > 700:
        step = 448
        for y in range(0, max(1, h - 640) + step, step):
            for x in range(0, max(1, w - 640) + step, step):
                y0, x0 = min(y, max(0, h - 640)), min(x, max(0, w - 640))
                b2, s2, k2 = _detect_once(sess, rgb[y0:y0 + 640, x0:x0 + 640], max(min_score, 0.6))
                for bb, sc_, kk in zip(b2, s2, k2):
                    boxes.append([bb[0] + x0, bb[1] + y0, bb[2], bb[3]])
                    scores.append(sc_)
                    kps.append(kk + np.array([x0, y0], np.float32))
    if not boxes:
        return []
    keep = np.array(cv2.dnn.NMSBoxes(boxes, scores, min_score, 0.3)).flatten()
    out = [{"box": tuple(boxes[i]), "lm": kps[i].astype(np.float32), "score": scores[i]} for i in keep]
    out.sort(key=lambda f: -f["box"][2] * f["box"][3])
    return out


def restore_faces(job, rgb, blend, base, span):
    faces = detect_faces(rgb, job)
    if not faces:
        job.add_note("No face was found, so faces were left as they are.")
        return rgb
    sess = session("face", job)
    name = sess.get_inputs()[0].name
    h, w = rgb.shape[:2]
    out = rgb.astype(np.float32)
    feather = np.zeros((512, 512), np.float32)
    feather[48:464, 48:464] = 1
    feather = cv2.GaussianBlur(feather, (0, 0), 22)
    faces = faces[:12]
    for i, f in enumerate(faces):
        job.upd(base + span * i / len(faces), f"Restoring face {i + 1} of {len(faces)}...")
        M, _ = cv2.estimateAffinePartial2D(f["lm"], FFHQ, method=cv2.LMEDS)
        if M is None:
            continue
        crop = cv2.warpAffine(rgb, M, (512, 512), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        x = ((crop.astype(np.float32) / 255 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        y = sess.run(None, {name: x})[0][0].transpose(1, 2, 0)
        y = np.clip((y + 1) / 2, 0, 1) * 255
        inv = cv2.invertAffineTransform(M)
        back = cv2.warpAffine(y, inv, (w, h), flags=cv2.INTER_CUBIC)
        mw = cv2.warpAffine(feather, inv, (w, h), flags=cv2.INTER_LINEAR)[..., None] * blend
        out = out * (1 - mw) + back * mw
    job.add_note(f"{len(faces)} face(s) restored.")
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- classic corrections
def est_noise(gray):
    g = gray.astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = g.shape
    if h < 8 or w < 8:
        return 0.0
    s = np.abs(cv2.filter2D(g, -1, k)[1:-1, 1:-1]).sum()
    return float(s * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2)))


def denoise(rgb, s):
    small = rgb if max(rgb.shape[:2]) <= 1200 else cv2.resize(rgb, None, fx=1200 / max(rgb.shape[:2]), fy=1200 / max(rgb.shape[:2]), interpolation=cv2.INTER_AREA)
    sig = est_noise(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
    if sig < 2.2:
        return rgb
    h = float(np.clip(sig * 0.8 * s, 1.5, 12))
    return tiled(lambda t: cv2.fastNlMeansDenoisingColored(t, None, h, h * 0.8, 7, 21), rgb)


def auto_tone(rgb, s):
    f = rgb.astype(np.float32) / 255
    mean = f.reshape(-1, 3).mean(0)
    gain = np.clip(mean.mean() / np.maximum(mean, 1e-3), 0.88, 1.14)  # gentle gray-world: fixes casts, not scenes
    f = np.clip(f * (1 + (gain - 1) * 0.75 * s), 0, 1)
    lum = f[::3, ::3] @ np.array([.299, .587, .114], np.float32)
    lo, hi = np.percentile(lum, [0.5, 99.5])
    lo, hi = min(lo, 0.10) * s, 1 - (1 - max(hi, 0.90)) * s
    f = np.clip((f - lo) / max(hi - lo, 0.2), 0, 1)
    m = float((f[::3, ::3] @ np.array([.299, .587, .114], np.float32)).mean())
    g = float(np.clip(np.log(0.46) / np.log(max(m, 0.05)), 0.7, 1.3))
    f = f ** (1 + (g - 1) * 0.8 * s)
    u8 = (f * 255 + 0.5).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    l2 = cv2.createCLAHE(clipLimit=1.2 + 1.3 * s, tileGridSize=(8, 8)).apply(lab[..., 0])
    k = 0.2 + 0.5 * s
    lab[..., 0] = (lab[..., 0] * (1 - k) + l2 * k + 0.5).astype(np.uint8)
    hsv = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255
    hsv[..., 1] = np.clip((sat + (1 - sat) * sat * 0.5 * s) * 255, 0, 255)  # vibrance: lifts dull colours, spares vivid ones
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))


def _guided(guide, src, r, eps):
    mi, mp = _box(guide, r), _box(src, r)
    a = (_box(guide * src, r) - mi * mp) / (_box(guide * guide, r) - mi * mi + eps)
    return _box(a, r) * guide + _box(mp - a * mi, r)


def haze_amount(rgb):
    """0 = clear photo, 1 = milky / faded (lifted blacks, washed-out contrast)."""
    g = cv2.cvtColor(rgb[::4, ::4], cv2.COLOR_RGB2GRAY)
    return float(np.clip((np.percentile(g, 1) / 255 - 0.04) / 0.2, 0, 1))


def dehaze(rgb, s):
    """Removes haze, fog and the washed-out 'faded' look (dark-channel prior, edge-aware)."""
    amt = haze_amount(rgb) * s
    if amt < 0.08:
        return rgb
    h, w = rgb.shape[:2]
    k = min(1.0, 900 / max(h, w))
    sm = cv2.resize(rgb, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else rgb
    f = sm.astype(np.float32) / 255
    sh, sw = f.shape[:2]
    patch = max(5, (min(sh, sw) // 70) | 1)
    ker = np.ones((patch, patch), np.uint8)
    dark = cv2.erode(f.min(2), ker)
    n = max(1, int(sh * sw * 0.002))
    A = np.minimum(f.reshape(-1, 3)[np.argpartition(dark.ravel(), -n)[-n:]].mean(0), 0.97)
    t = 1 - min(0.85, 0.85 * amt) * cv2.erode((f / A).min(2), ker)
    gray = cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    t = np.clip(_guided(gray, t.astype(np.float32), patch * 3, 1e-3), 0.35, 1)
    if k < 1:
        t = cv2.resize(t, (w, h), interpolation=cv2.INTER_CUBIC)
    out = (rgb.astype(np.float32) / 255 - A) / t[..., None] + A
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def deblur(rgb, s):
    """Undoes soft focus / camera blur on the brightness channel (Richardson-Lucy), so nothing is invented."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sm = g if max(g.shape) <= 1000 else cv2.resize(g, None, fx=1000 / max(g.shape), fy=1000 / max(g.shape), interpolation=cv2.INTER_AREA)
    lv = cv2.Laplacian(sm, cv2.CV_64F).var()
    if lv > 700:
        return rgb  # already crisp
    sigma = (1.7 if lv < 40 else 1.4 if lv < 120 else 1.0) * max(1.0, max(g.shape) / 1300)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] / 255 + 1e-3
    r = int(np.ceil(3 * sigma))
    kk = cv2.getGaussianKernel(2 * r + 1, sigma)
    psf = (kk @ kk.T).astype(np.float32)
    est = L.copy()
    for _ in range(10):
        conv = cv2.filter2D(est, -1, psf, borderType=cv2.BORDER_REFLECT)
        est = np.clip(est * cv2.filter2D(L / np.maximum(conv, 1e-3), -1, psf, borderType=cv2.BORDER_REFLECT), 0, 1)
    a = min(1.0, 0.9 * s)
    lab[..., 0] = np.clip((L * (1 - a) + est * a) * 255, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def sharpen(rgb, s):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sm = g if max(g.shape) <= 1000 else cv2.resize(g, None, fx=1000 / max(g.shape), fy=1000 / max(g.shape), interpolation=cv2.INTER_AREA)
    sharp = cv2.Laplacian(sm, cv2.CV_64F).var()
    boost = 1.0 if sharp > 300 else 1.5 if sharp > 80 else 2.0  # softer photos get more help
    amount = min(1.4, 0.5 * s * boost)
    sigma = max(0.8, min(rgb.shape[:2]) / 1600) * 1.3
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    detail = L - cv2.GaussianBlur(L, (0, 0), sigma)
    lab[..., 0] = np.clip(L + detail * amount * smooth(np.abs(detail), 2.0, 6.0), 0, 255)  # skips flat areas so noise stays quiet
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------- super resolution
def esrgan(job, rgb, scale, base, span):
    """Natural-looking 4x super resolution (SPAN). 2x is made by shrinking the 4x result, which keeps it crisp."""
    sess = session("upscale", job)
    name = sess.get_inputs()[0].name
    H, W = rgb.shape[:2]
    tile, pad, k = 256, 16, 4
    out = np.empty((H * k, W * k, 3), np.uint8)
    spots = [(x, y) for y in range(0, H, tile) for x in range(0, W, tile)]
    for i, (x, y) in enumerate(spots):
        job.upd(base + span * i / len(spots), f"Upscaling {scale}x... {int(100 * i / len(spots))}%")
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + tile + pad), min(H, y + tile + pad)
        t = (rgb[y0:y1, x0:x1].astype(np.float32) / 255).transpose(2, 0, 1)[None]
        r = np.clip(sess.run(None, {name: t})[0][0].transpose(1, 2, 0), 0, 1)
        cw, ch = min(tile, W - x) * k, min(tile, H - y) * k
        cx, cy = (x - x0) * k, (y - y0) * k
        out[y * k:y * k + ch, x * k:x * k + cw] = (r[cy:cy + ch, cx:cx + cw] * 255 + 0.5).astype(np.uint8)
    if scale == 2:
        out = cv2.resize(out, (W * 2, H * 2), interpolation=cv2.INTER_AREA)
    return out


def upscale(job, rgb, scale, base, span):
    long = max(rgb.shape[:2])
    if long * scale > MAX_SIDE:
        f = MAX_SIDE / (long * scale)
        rgb = cv2.resize(rgb, (max(1, int(rgb.shape[1] * f)), max(1, int(rgb.shape[0] * f))), interpolation=cv2.INTER_AREA)
        job.add_note(f"The result is capped at {MAX_SIDE}px on the long side.")
    return esrgan(job, rgb, scale, base, span)


# ---------------------------------------------------------------- inpainting
def _telea(rgb, m):
    return cv2.inpaint(rgb, (m > 0).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)


def lama_fill(job, rgb, mask, base=0, span=100):
    """Remove whatever is under the mask (white = remove) and rebuild the background."""
    m = (mask > 127).astype(np.uint8)
    if not m.any():
        return rgb
    m = cv2.dilate(m, np.ones((7, 7), np.uint8))
    try:
        sess = session("lama", job)
    except LabError as e:
        job.add_note("The AI object remover is not available, so a basic fill was used. " + str(e))
        return _telea(rgb, m)
    h, w = rgb.shape[:2]
    pad = 512
    canvas = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    todo = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    refl = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)  # mirrored copies of the hole in the padding must be holes too
    refl[pad:pad + h, pad:pad + w] = 0
    n, lab = cv2.connectedComponents(cv2.dilate(todo, np.ones((61, 61), np.uint8)))
    total = max(1, n - 1)
    for k in range(1, n):
        job.upd(base + span * (k - 1) / total, f"Removing object {k} of {total}...")
        reg = (lab == k) & (todo > 0)
        if not reg.any():
            continue
        ys, xs = np.nonzero(reg)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        side = max(y1 - y0, x1 - x0) + 1
        win = int(min(max(512, side * 1.7), min(canvas.shape[:2])))
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        wy = int(np.clip(cy - win // 2, 0, canvas.shape[0] - win))
        wx = int(np.clip(cx - win // 2, 0, canvas.shape[1] - win))
        img_w = canvas[wy:wy + win, wx:wx + win]
        msk_w = todo[wy:wy + win, wx:wx + win].astype(np.float32)
        a = cv2.resize(img_w, (512, 512), interpolation=cv2.INTER_AREA if win > 512 else cv2.INTER_CUBIC)
        b = (cv2.resize(np.maximum(msk_w, refl[wy:wy + win, wx:wx + win].astype(np.float32)), (512, 512), interpolation=cv2.INTER_AREA) > 0.2).astype(np.float32)
        r = sess.run(None, {"image": (a.astype(np.float32) / 255).transpose(2, 0, 1)[None], "mask": b[None, None]})[0][0]
        r = np.clip(r.transpose(1, 2, 0), 0, 255)
        if win != 512:
            r = cv2.resize(r, (win, win), interpolation=cv2.INTER_CUBIC)
        alpha = cv2.GaussianBlur(msk_w, (0, 0), 2.0)[..., None]
        alpha = np.maximum(alpha, msk_w[..., None] * 0.999)
        canvas[wy:wy + win, wx:wx + win] = np.clip(img_w * (1 - alpha) + r * alpha + 0.5, 0, 255).astype(np.uint8)
        todo[wy:wy + win, wx:wx + win] = 0
    return canvas[pad:pad + h, pad:pad + w]


def repair_damage(job, rgb, s, base, span):
    """Find thin scratches and dust specks in smooth areas (never real edges or faces) and fill them in."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    k = max(5, (min(h, w) // 300) | 1)
    med = cv2.medianBlur(g, k)  # removes thin lines and specks but keeps real edges
    T = 30 - 12 * s
    cand = (np.abs(g.astype(np.int16) - med.astype(np.int16)) > T).astype(np.uint8)
    zone = cv2.dilate(cv2.Canny(med, 40, 110), np.ones((k + 4, k + 4), np.uint8)) > 0  # areas with real structure
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep = np.zeros_like(cand)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        long = max(bw, bh)
        line = long >= 6 * k and area / long <= 0.9 * k          # long and thin: a scratch
        speck = area <= 4 * k * k and max(bw, bh) <= 3 * k        # tiny: dust
        if not (line or speck):
            continue
        comp = lab[y:y + bh, x:x + bw] == i
        if zone[y:y + bh, x:x + bw][comp].mean() > (0.6 if line else 0.35):
            continue                                            # sits on real detail: leave it alone
        keep[y:y + bh, x:x + bw][comp] = 1
    for f in detect_faces(rgb, job):  # never "repair" real facial detail
        x, y, fw, fh = f["box"]
        keep[max(0, int(y - fh * .2)):int(y + fh * 1.2), max(0, int(x - fw * .2)):int(x + fw * 1.2)] = 0
    if keep.sum() < 3:
        job.add_note("No scratches or dust were found.")
        return rgb
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8))
    return lama_fill(job, rgb, keep * 255, base, span)


# ---------------------------------------------------------------- colorize
def colorize(job, rgb, base, span):
    sess = session("color", job)
    job.upd(base + span * .3, "Colorizing...")
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB).astype(np.float32) / 255
    L = cv2.cvtColor(gray, cv2.COLOR_RGB2LAB)[..., :1]
    small = cv2.cvtColor(cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB)
    small[..., 1:] = 0
    inp = cv2.cvtColor(small, cv2.COLOR_LAB2RGB).transpose(2, 0, 1)[None].astype(np.float32)
    ab = sess.run(None, {sess.get_inputs()[0].name: inp})[0][0].transpose(1, 2, 0)
    ab = cv2.resize(ab, (w, h), interpolation=cv2.INTER_CUBIC) * 0.88  # slightly calmer colours look more natural
    out = cv2.cvtColor(np.concatenate([L, ab], -1).astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- retouch
def _face_masks(rgb, f):
    """Soft mask of this person's skin (colour-matched to their own cheeks), minus eyes, brows and mouth."""
    h, w = rgb.shape[:2]
    x, y, fw, fh = f["box"]
    lm = f["lm"]
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int(x + fw / 2), int(y + fh * .56)), (int(fw * .62), int(fh * .78)), 0, 0, 360, 1.0, -1)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    r = max(2, int(fw * .07))
    pts = [lm[2], (lm[0] + lm[3]) / 2, (lm[1] + lm[4]) / 2]  # nose and both cheeks
    samp = np.concatenate([lab[max(0, int(py) - r):int(py) + r, max(0, int(px) - r):int(px) + r].reshape(-1, 3) for px, py in pts])
    ref = np.median(samp, 0)
    d = np.sqrt((0.8 * (lab[..., 0] - ref[0])) ** 2 + (lab[..., 1] - ref[1]) ** 2 + (lab[..., 2] - ref[2]) ** 2)
    m *= 1 - smooth(d, 12, 28)  # hair, lips, shadows and background drop out
    for ex, ey in lm[:2]:
        cv2.circle(m, (int(ex), int(ey)), int(fw * .17), 0, -1)
        cv2.ellipse(m, (int(ex), int(ey - fw * .13)), (int(fw * .2), int(fw * .1)), 0, 0, 360, 0, -1)  # brows
    mc = lm[3:5].mean(0)
    mw = float(np.linalg.norm(lm[3] - lm[4]))
    cv2.ellipse(m, (int(mc[0]), int(mc[1])), (int(mw * .75), int(mw * .4)), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), max(1.0, fw * .025)), mc, mw


def retouch(job, rgb, p):
    faces = detect_faces(rgb, job)
    if not faces:
        raise LabError("No face was found. Retouching needs a clear, front-facing face.")
    skin_amt, eyes_amt, teeth_amt = (float(p.get(k, 0)) for k in ("skin", "eyes", "teeth"))
    spots = bool(p.get("blemish", False))
    out = rgb.copy()
    for i, f in enumerate(faces[:8]):
        job.upd(10 + 85 * i / min(len(faces), 8), f"Retouching face {i + 1}...")
        x, y, fw, fh = f["box"]
        pad = int(fw * .5)
        H, W = rgb.shape[:2]
        y0, y1, x0, x1 = max(0, int(y - pad)), min(H, int(y + fh + pad)), max(0, int(x - pad)), min(W, int(x + fw + pad))
        roi = out[y0:y1, x0:x1].copy()
        f2 = dict(f, box=(x - x0, y - y0, fw, fh), lm=f["lm"] - np.array([x0, y0], np.float32))
        m, mc, mw = _face_masks(roi, f2)
        if spots:
            gl = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            k = max(5, int(fw * .05) | 1)
            bh = cv2.morphologyEx(gl, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            cand = ((bh > 14) & (m > .6)).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(cand)
            fix = np.zeros_like(cand)
            for j in range(1, n):
                if 3 <= st[j][4] <= (fw * .06) ** 2 * 3.14:
                    fix[lab == j] = 1
            if fix.any():
                roi = cv2.inpaint(roi, cv2.dilate(fix, np.ones((5, 5), np.uint8)) * 255, 3, cv2.INPAINT_TELEA)
        if skin_amt > 0:
            fl = roi.astype(np.float32)
            r = max(2.0, fw * .018)
            low = cv2.GaussianBlur(fl, (0, 0), r * 1.6)
            sm = cv2.bilateralFilter(low, 0, 16 + 34 * skin_amt, r * 2.2)
            res = sm + (fl - low) * (0.7 - 0.4 * skin_amt)  # keeps some pore texture so skin stays real
            a = (m * skin_amt)[..., None]
            roi = np.clip(fl * (1 - a) + res * a, 0, 255).astype(np.uint8)
        if eyes_amt > 0 or teeth_amt > 0:
            lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB).astype(np.float32)
            if eyes_amt > 0:
                for ex, ey in f2["lm"][:2]:
                    rr = max(3, int(fw * .11))
                    yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
                    g = np.exp(-((xx - ex) ** 2 + (yy - ey) ** 2) / (2 * (rr * .8) ** 2)).astype(np.float32)
                    lab[..., 0] += g * 12 * eyes_amt
            if teeth_amt > 0:
                tm = np.zeros(roi.shape[:2], np.float32)
                cv2.ellipse(tm, (int(mc[0]), int(mc[1])), (int(mw * .45), int(mw * .2)), 0, 0, 360, 1.0, -1)
                tm = cv2.GaussianBlur(tm, (0, 0), max(1.0, mw * .04))
                is_teeth = smooth(lab[..., 0], 130, 175) * (1 - smooth(lab[..., 1], 134, 146))  # bright and not red (OpenCV LAB: a offset 128)
                wt = tm * is_teeth * teeth_amt
                lab[..., 2] = lab[..., 2] - (lab[..., 2] - 128) * 0.55 * wt
                lab[..., 0] += 14 * wt
            roi = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        out[y0:y1, x0:x1] = roi
    return out


# ---------------------------------------------------------------- background blur
bg_mask = None  # set by app.py: fn(rgb) -> float mask 0..1, subject = 1


def _disk(r):
    k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(k, (r, r), r, 1.0, -1)
    return k / k.sum()


def _blur_masked(img, w, r, lens):
    """Blur using only background pixels so the subject's colours never bleed into the blur."""
    s = min(1.0, 24.0 / max(r, 1)) if lens else 1.0
    f, ww = img * w[..., None], w
    if s < 1:
        f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ww = cv2.resize(ww, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rr = max(1, int(round(r * s)))
    if lens:
        k = _disk(rr)
        num, den = cv2.filter2D(f, -1, k), cv2.filter2D(ww, -1, k)
    else:
        num, den = cv2.GaussianBlur(f, (0, 0), max(rr / 2, .5)), cv2.GaussianBlur(ww, (0, 0), max(rr / 2, .5))
    res = num / np.maximum(den[..., None], 1e-3)
    if s < 1:
        res = cv2.resize(res, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return res


def blur_background(job, rgb, p):
    if bg_mask is None:
        raise LabError("The background model is not available.")
    job.upd(8, "Finding the subject (the first use downloads a model)...")
    m = bg_mask(rgb).astype(np.float32)
    job.upd(60, "Blurring the background...")
    h, w = m.shape
    r = max(2, int(float(p.get("amount", .5)) * 0.03 * max(h, w)))
    lens = p.get("style") == "lens"
    f = rgb.astype(np.float32)
    mm = cv2.GaussianBlur(m, (0, 0), 1.2)
    bgw = np.clip(1 - m * 1.15, 0, 1)
    far = _blur_masked(f, bgw, r, lens)
    near = _blur_masked(f, bgw, max(1, int(r * .35)), lens)
    d = cv2.distanceTransform((m < .5).astype(np.uint8), cv2.DIST_L2, 5)
    t = np.clip(d / (0.12 * max(h, w)), 0, 1)[..., None]  # blur grows with distance from the subject
    bgimg = near * (1 - t) + far * t
    out = f * mm[..., None] + bgimg * (1 - mm[..., None])
    return np.clip(out + .5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- privacy
def privacy(job, rgb, mask, p):
    m = np.zeros(rgb.shape[:2], np.float32)
    nfaces = 0
    if p.get("faces", True):
        job.upd(10, "Finding faces...")
        for f in detect_faces(rgb, job, 0.6):
            x, y, fw, fh = f["box"]
            cv2.ellipse(m, (int(x + fw / 2), int(y + fh * .52)), (int(fw * .72), int(fh * .85)), 0, 0, 360, 1.0, -1)
            nfaces += 1
    if mask is not None:
        m = np.maximum(m, (mask > 127).astype(np.float32))
    if not m.any():
        raise LabError("Nothing to hide. Turn on face blur or paint over the area first.")
    job.upd(60, "Hiding...")
    ys, xs = np.nonzero(m > 0)
    size = max(ys.max() - ys.min(), xs.max() - xs.min(), 1)
    f = rgb.astype(np.float32)
    if p.get("mode") == "pixelate":
        cells = max(8, int(max(rgb.shape[:2]) / 36))
        sm = cv2.resize(rgb, (max(1, rgb.shape[1] // cells), max(1, rgb.shape[0] // cells)), interpolation=cv2.INTER_AREA)
        hid = cv2.resize(sm, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    else:
        hid = cv2.GaussianBlur(f, (0, 0), max(6, max(rgb.shape[:2]) * .012))
        hid = cv2.GaussianBlur(hid, (0, 0), max(6, max(rgb.shape[:2]) * .012))
    mm = cv2.GaussianBlur(m, (0, 0), 1.5)[..., None]
    if nfaces:
        job.add_note(f"{nfaces} face(s) hidden.")
    return np.clip(f * (1 - mm) + hid * mm + .5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- tools
def _plan(job, steps, img):
    """steps: [(label, weight, fn(img, base, span) -> img)]. Runs them in order with one progress bar."""
    total = float(sum(w for _, w, _ in steps)) or 1.0
    done = 0.0
    for label, wt, fn in steps:
        base, span = 100 * done / total, 100 * wt / total
        job.upd(base, label + "...")
        img = fn(img, base, span)
        done += wt
    job.upd(99, "Finishing...")
    return img


def _prefetch(job, keys):
    for k in keys:
        ensure(k, job)


def tool_enhance(job, img, mask, p):
    s = {"natural": .65, "strong": 1.0}.get(p.get("strength"), .8)
    up = int(p.get("upscale") or 0)
    faces = bool(p.get("faces", True))
    _prefetch(job, (["upscale"] if up in (2, 4) else []) + (["detect", "face"] if faces else []))
    hazy = [False]

    def do_dehaze(a, b, c):
        hazy[0] = haze_amount(a) > 0.08
        return dehaze(a, s)

    steps = []
    if p.get("denoise", True):
        steps.append(("Reducing noise", 2, lambda a, b, c: denoise(a, s)))
    if p.get("dehaze", True):
        steps.append(("Removing haze and fading", 1, do_dehaze))
    if p.get("deblur", True):
        steps.append(("Removing blur", 2, lambda a, b, c: deblur(a, s)))
    if up in (2, 4):
        steps.append((f"Upscaling {up}x", 3, lambda a, b, c: upscale(job, a, up, b, c)))
    if faces:
        steps.append(("Restoring faces", 5, lambda a, b, c: restore_faces(job, a, .85 if s < .8 else 1.0, b, c)))
    if p.get("tone", True):
        steps.append(("Fixing light and colour", 1, lambda a, b, c: auto_tone(a, s * (.35 if hazy[0] else 1.0))))
    if p.get("sharpen", True):
        steps.append(("Sharpening", 1, lambda a, b, c: sharpen(a, s * .6)))
    if not steps:
        raise LabError("Pick at least one improvement.")
    return to_png(_plan(job, steps, img)), "image/png"


def tool_upscale(job, img, mask, p):
    up = 4 if int(p.get("scale", 2)) == 4 else 2
    faces = bool(p.get("faces", False))
    _prefetch(job, ["upscale"] + (["detect", "face"] if faces else []))
    steps = [(f"Upscaling {up}x", 3, lambda a, b, c: upscale(job, a, up, b, c))]
    if faces:
        steps.append(("Restoring faces", 5, lambda a, b, c: restore_faces(job, a, .75, b, c)))
    return to_png(_plan(job, steps, img)), "image/png"


def tool_erase(job, img, mask, p):
    if mask is None or not (mask > 127).any():
        raise LabError("Paint over the object you want to remove first.")
    try:
        ensure("lama", job)
    except LabError as e:
        job.add_note(str(e))
    return to_png(lama_fill(job, img, mask, 0, 100)), "image/png"


def tool_restore(job, img, mask, p):
    s = {"natural": .6, "strong": 1.0}.get(p.get("strength"), .8)
    up = int(p.get("upscale") or 0)
    faces, col, scr = bool(p.get("faces", True)), bool(p.get("colorize", False)), bool(p.get("scratches", True))
    _prefetch(job, (["detect", "lama"] if scr else []) + (["color"] if col else []) + (["upscale"] if up in (2, 4) else [])
              + (["detect", "face"] if faces else []))
    steps = []
    if scr:
        steps.append(("Repairing scratches and dust", 4, lambda a, b, c: repair_damage(job, a, s, b, c)))
    steps.append(("Reducing grain", 2, lambda a, b, c: denoise(a, s)))
    if col:
        steps.append(("Colorizing", 6, lambda a, b, c: colorize(job, a, b, c)))
    if up in (2, 4):
        steps.append((f"Upscaling {up}x", 3, lambda a, b, c: upscale(job, a, up, b, c)))
    steps.append(("Removing blur", 2, lambda a, b, c: deblur(a, s)))
    if faces:
        steps.append(("Restoring faces", 5, lambda a, b, c: restore_faces(job, a, .9, b, c)))
    if p.get("tone", True):
        steps.append(("Fixing fading and contrast", 1, lambda a, b, c: auto_tone(dehaze(a, s), s * .6)))
    steps.append(("Sharpening", 1, lambda a, b, c: sharpen(a, s * .8)))
    return to_png(_plan(job, steps, img)), "image/png"


def tool_retouch(job, img, mask, p):
    _prefetch(job, ["detect"])
    return to_png(retouch(job, img, p)), "image/png"


def tool_blurbg(job, img, mask, p):
    return to_png(blur_background(job, img, p)), "image/png"


def tool_privacy(job, img, mask, p):
    _prefetch(job, ["detect"])
    return to_png(privacy(job, img, mask, p)), "image/png"


VT = {
    "logo": dict(colormode="color", hierarchical="stacked", mode="spline", filter_speckle=8, color_precision=5, layer_difference=24,
                 corner_threshold=60, length_threshold=4.0, splice_threshold=45, path_precision=3),
    "illustration": dict(colormode="color", hierarchical="stacked", mode="spline", filter_speckle=4, color_precision=7, layer_difference=12,
                         corner_threshold=60, length_threshold=4.0, splice_threshold=45, path_precision=3),
    "bw": dict(colormode="binary", hierarchical="stacked", mode="spline", filter_speckle=6, color_precision=6, layer_difference=16,
               corner_threshold=60, length_threshold=4.0, splice_threshold=45, path_precision=3),
}


def tool_vectorize(job, img, mask, p):
    if vtracer is None:
        raise LabError("Vectorizing needs one more package. Run: pip install vtracer")
    job.upd(20, "Tracing shapes...")
    long = max(img.shape[:2])
    if long > 1800:
        img = cv2.resize(img, None, fx=1800 / long, fy=1800 / long, interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    svg = vtracer.convert_raw_image_to_svg(buf.getvalue(), img_format="png", **VT.get(p.get("preset"), VT["logo"]))
    return svg.encode("utf-8"), "image/svg+xml"


TOOLS = {"enhance": tool_enhance, "upscale": tool_upscale, "erase": tool_erase, "restore": tool_restore,
         "retouch": tool_retouch, "blurbg": tool_blurbg, "privacy": tool_privacy, "vectorize": tool_vectorize}


# ---------------------------------------------------------------- routes
@router.get("/models")
def models():
    return models_status()


@router.post("/run")
def run(tool: str = Form(...), file: UploadFile = File(...), mask: UploadFile = File(None), params: str = Form("{}")):
    if tool not in TOOLS:
        raise HTTPException(400, "Unknown tool")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "That image is larger than 40 MB.")
    try:
        p = json.loads(params) or {}
    except Exception:
        p = {}
    try:
        img = load_rgb(data)
        m = load_mask(mask.file.read(), img.shape[:2]) if mask is not None else None
    except LabError as e:
        raise HTTPException(400, str(e))
    _prune()
    job = Job()
    with _jobs_lock:
        _jobs[job.id] = job
    threading.Thread(target=_worker, args=(job, TOOLS[tool], img, m, p), daemon=True).start()
    return {"job": job.id}


@router.get("/job/{jid}")
def job_status(jid: str):
    j = _jobs.get(jid)
    if not j:
        raise HTTPException(404, "Unknown job")
    return {"state": j.state, "pct": j.pct, "msg": j.msg, "error": j.error, "note": j.note}


@router.get("/result/{jid}")
def job_result(jid: str):
    j = _jobs.get(jid)
    if not j or j.state != "done":
        raise HTTPException(404, "Result not ready")
    return Response(j.data, media_type=j.mime)
