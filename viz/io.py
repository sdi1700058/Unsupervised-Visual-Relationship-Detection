"""viz/io.py — figure writer.

`save_with_caption()` keeps the `what/why/how_to_read` arguments for caller
back-compat but writes ONLY the PNG by default. The earlier per-figure
`<name>.caption.md` sidecar was retracted by the user — it created hundreds of
stray files and made the viz dir hard to navigate. Re-enable with
`VIZ_CAPTIONS=1` env var if ever needed.
"""

import os


def save_with_caption(fig, path, *, what=None, why=None, how_to_read=None, dpi=150):
    """Save `fig` to `<path>.png`. Returns (png_path, caption_path_or_None).

    Captions are NOT written unless `VIZ_CAPTIONS=1` is set in the env.
    """
    if not path.endswith(".png"):
        path = path + ".png"

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass

    caption_path = None
    if os.environ.get("VIZ_CAPTIONS") == "1" and (what or why or how_to_read):
        caption_path = path[:-4] + ".caption.md"
        with open(caption_path, "w") as f:
            f.write(f"# {os.path.basename(path)}\n\n")
            if what:        f.write(f"## What\n{what}\n\n")
            if why:         f.write(f"## Why it matters\n{why}\n\n")
            if how_to_read: f.write(f"## How to read\n{how_to_read}\n")

    return path, caption_path
