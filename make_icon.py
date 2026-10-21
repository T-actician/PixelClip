"""Create pixelclip.ico (dashed pink square on dark tile). Run: python make_icon.py"""
from PIL import Image, ImageDraw

S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle((0, 0, S - 1, S - 1), radius=int(S * 0.22), fill="#0f1b22")
a, b, w, r = int(S * .22), int(S * .78), int(S * .08), int(S * .125)
pink, dash, gap = "#ff4d8d", 130, 85


def seg(p0, p1):  # dashed straight line
    (x0, y0), (x1, y1) = p0, p1
    L = max(abs(x1 - x0), abs(y1 - y0)); t = 0
    while t < L:
        e = min(t + dash, L); k0, k1 = t / L, e / L
        d.line((x0 + (x1 - x0) * k0, y0 + (y1 - y0) * k0, x0 + (x1 - x0) * k1, y0 + (y1 - y0) * k1), fill=pink, width=w)
        t += dash + gap


seg((a + r, a), (b - r, a)); seg((b, a + r), (b, b - r))
seg((b - r, b), (a + r, b)); seg((a, b - r), (a, a + r))
for cx, cy, st in ((b - r, a + r, 270), (b - r, b - r, 0), (a + r, b - r, 90), (a + r, a + r, 180)):
    d.arc((cx - r, cy - r, cx + r, cy + r), st, st + 90, fill=pink, width=w)
img.save("pixelclip.ico", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
print("Created pixelclip.ico")
