# PixelClip

Fast, clean image background removal that runs entirely on your own computer. No uploads, no accounts, no limits.

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

## Quick start

Requires Python 3.9 or newer.

```bash
git clone https://github.com/<your-username>/PixelClip.git
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
6. Choose a format, then **Download image** or **Download all as ZIP**.

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
  app.py            FastAPI server and background removal
  index.html        The whole interface (HTML, CSS, JS)
  build.py          Standalone app builder
  requirements.txt  Python dependencies
```

## Notes

- Everything runs locally. Images are never sent anywhere except from your browser to the app on your own machine. The "From a link" feature downloads the image you point to.
- Images larger than 4096 px on the long side are scaled down.
- It runs on CPU by default. For an NVIDIA GPU, replace `rembg[cpu]` with `rembg[gpu]` in `requirements.txt`.
- Shadow and outline effects in exports need a browser that supports canvas filters (Chrome, Edge, and Firefox do; Safari does not yet).

## Built with

[rembg](https://github.com/danielgatis/rembg), [FastAPI](https://fastapi.tiangolo.com/), [Pillow](https://python-pillow.org/), and [ONNX Runtime](https://onnxruntime.ai/).

## License

Add a license of your choice (MIT is a common pick) before sharing publicly.
