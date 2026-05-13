"""viz/io.py — figure + caption writer per SPEC §V6 / C12.

Every public figure goes through `save_with_caption()`. Producing a
`.png` without its `.caption.md` sibling is a contract violation.
"""

import os


def save_with_caption(fig, path, *, what, why, how_to_read, dpi=150):
    """Save `fig` to `<path>` AND write `<path-without-ext>.caption.md`.

    Parameters
    ----------
    fig         : matplotlib.figure.Figure
    path        : str — absolute or workspace-relative target path; if it
                  doesn't end in `.png` we append it.
    what        : str — one-sentence description of what the image shows.
    why         : str — one-sentence justification ("why we care").
    how_to_read : str — one-sentence reader instructions.
    dpi         : int — figure resolution.

    Returns the two written paths as (png_path, caption_path).
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

    caption_path = path[:-4] + ".caption.md"
    name = os.path.basename(path)
    with open(caption_path, "w") as f:
        f.write(f"# {name}\n\n")
        f.write(f"## What\n{what}\n\n")
        f.write(f"## Why it matters\n{why}\n\n")
        f.write(f"## How to read\n{how_to_read}\n")

    return path, caption_path
