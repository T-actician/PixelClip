# PixelClip

Fast, clean image background removal plus everyday photo tools (ID photos, product and social sizes, watermarking, file-size limits) that run entirely on your own computer. No uploads, no accounts, no limits.

Built for designers and developers who need cutouts without sending their work to a website.

## Features

- **One-command start:** `python app.py` opens the app in your browser
- **Drag, drop, or paste:** add images by dropping them, pasting with Ctrl+V, or pasting a link
- **Batch processing:** queue many images and download them all as a ZIP
- **Three quality levels:** Fast, Balanced, and Ultra (sharpest hair and fur edges)
- **Soften edges:** optional alpha matting for fine hair and glass
- **Before/after slider:** drag across the image to compare
- **Touch-up brushes:** Erase and Restore with adjustable size, plus Undo, Redo, and Reset
- **Backgrounds:** transparent, preset colors, a custom color, or your own image
- **Effects:** drop shadow and colored outline
- **Export formats:** PNG, WebP, or JPG
- **Passport / ID photos:** 35x45 mm and 2x2 in at 300 dpi. Finds the face, sizes and centers the head, adds a white background, and can lay out a 4x6 in or A4 print sheet with many copies
- **Product presets:** marketplace square, high-res square, and 4:5 portrait, with the subject centered and padded on white
- **Social sizes:** Instagram, Story/WhatsApp status, profile picture, Facebook cover, X header, YouTube thumbnail, LinkedIn banner, or any custom size. Choose Subject, Fit, or Fill
- **Watermark:** text or logo, any corner, centered, or tiled, with size and opacity. Applied to every image, including the ZIP
- **Limit file size:** set a target such as 100 KB. Quality drops first, then dimensions, until it fits
- **Output view:** shows exactly what you will download, with size, format, and file weight

### Photo Lab (open it from the **Photo Lab** tab, or at `/lab`)
- **Enhance:** one click for noise reduction, light and colour correction, sharpening, face restoration, and optional 2x/4x upscale
- **Upscale:** 2x or 4x, natural-looking, no blocky edges
- **Erase objects:** paint over a person, wire, sign or blemish and it is rebuilt from the surroundings
- **Restore old photo:** repairs scratches and dust, reduces grain, restores faces, fixes fading, and can colorize black and white
- **Portrait retouch:** skin smoothing that keeps texture, blemish removal, brighter eyes, whiter teeth
- **Blur background:** soft or lens-style (bokeh) blur that grows with distance from the subject
- **Hide faces / areas:** auto-blur or pixelate every face, plus anything you paint over (number plates, documents)
- **Vectorize:** turn a logo or drawing into a scalable SVG
- Every tool supports **Apply to all photos**, **Undo**, and a before/after slider. Finished photos can be sent straight to **Cutout**

AI models download once, the first time a tool needs them, and are stored in `~/.pixelclip/models` (set `PIXELCLIP_MODELS` to change the folder). Sizes: face restorer 340 MB, object remover 208 MB, colorizer 980 MB, upscaler 2 MB, face finder 1 MB. Only one large model is kept in memory at a time.

## Quick start

Requires Python 3.9 or newer.

```bash
git clone https://github.com/<T-actician>/PixelClip.git
cd PixelClip

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Your browser opens at `http://127.0.0.1:8765`.

The first time you use each quality level, its AI model downloads once and is then cached on your machine.

## How to use

1. Add images: drop them on the page, click **Choose images**, paste with Ctrl+V, or paste a link under **From a link**.
2. Pick a **Quality** level. Use **Ultra** for portraits and animals.
3. Drag the slider in **Compare** view to check the result.
4. Switch to **Touch up** to fix mistakes with **Erase** and **Restore**.
5. Pick a background and any effects.
6. Optional: choose a **Size & layout** preset, add a **Watermark**, or turn on **Limit file size**. Open **Output** to preview the result.
7. Choose a format, then **Download image** or **Download all as ZIP**.

### Shortcuts

| Keys | Action |
| --- | --- |
| Ctrl+Z | Undo brush stroke |
| Ctrl+Shift+Z or Ctrl+Y | Redo |
| Ctrl+V | Paste an image |

## Quality levels

| Level | Model | Best for |
| --- | --- | --- |
| Fast | u2netp | Quick previews, large batches |
| Balanced | isnet-general-use | Everyday products and objects |
| Ultra | birefnet-general | Hair, fur, fine detail (slower, larger download) |

## Build a standalone app

Create a version that runs without Python installed:

```bash
pip install pyinstaller
python build.py
```

The result is in `dist/PixelClip/`. Share that whole folder.

## Project structure

```
PixelClip/
  app.py            FastAPI server, background removal, face finder
  lab.py            Photo Lab engine (enhance, upscale, erase, restore, retouch, blur, privacy, vectorize)
  lab.html          Photo Lab page
  index.html        The whole interface (HTML, CSS, JS)
  build.py          Standalone app builder
  requirements.txt  Python dependencies
```

## Notes

- Everything runs locally. Images are never sent anywhere except from your browser to the app on your own machine. The "From a link" feature downloads the image you point to.
- Images larger than 4096 px on the long side are scaled down.
- Passport / ID photos need a front-facing face. If none is found, framing is estimated from the cutout.
- Print sheets and ID sizes are exported at 300 dpi pixel dimensions.
- It runs on CPU by default. For an NVIDIA GPU, replace `rembg[cpu]` with `rembg[gpu]` in `requirements.txt`.
- Shadow and outline effects in exports need a browser that supports canvas filters (Chrome, Edge, and Firefox do; Safari does not yet).

## Built with

[rembg](https://github.com/danielgatis/rembg), [FastAPI](https://fastapi.tiangolo.com/), [Pillow](https://python-pillow.org/), and [ONNX Runtime](https://onnxruntime.ai/).

## License

Add a license of your choice (MIT is a common pick) before sharing publicly.
