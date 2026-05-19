"""Shared path utilities for the FOSAE project.

Composes the canonical hierarchical output directory consumed by
strips.py and sh/submit.sh (single source of truth — see SPEC §C15, §V13).

Layout: ``out/<domain>/<category>/<run_tag>/`` (labeled_objects is 2-tier:
``out/labeled_objects/<run_tag>/`` — no category).
"""

import os
import json
import hashlib

# Resolve project root relative to this file: latplan/util/paths.py -> project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# OUT_DIR is env-overridable so concurrent runs can use separate output trees
# (each gets its own out/grid_search.log — the tuning log is shared otherwise
# and a single logged trial + limit=1 makes later runs short-circuit training).
OUT_DIR  = os.environ.get("OUT_DIR") or os.path.join(PROJECT_ROOT, "out")


def find_dataset(filename):
    """Look for dataset in data/ first, fall back to latplan/puzzles/."""
    import latplan
    for p in (os.path.join(DATA_DIR, "npz", filename),
              os.path.join(DATA_DIR, filename)):
        if os.path.exists(p):
            return p
    return os.path.join(latplan.__path__[0], "puzzles", filename)


def domain_category(domain, **kw):
    """Map a strips.py task name -> (domain_seg, category_seg) for the out path.

    Categories follow the user-locked schema:
      puzzle       -> ('puzzle', '<type>-<width>-<height>')
      blocks       -> ('blocks', '<track>')
      vidvrd       -> ('video',  'vidvrd')
      actiongenome -> ('video',  'actiongenome')
      labeled_objects -> ('labeled_objects', None)   # 2-tier
    """
    if domain == "puzzle":
        return "puzzle", "{}-{}-{}".format(kw["type"], kw["width"], kw["height"])
    if domain in ("blocks", "blocksworld"):
        return "blocks", kw["track"]
    if domain in ("vidvrd", "actiongenome"):
        return "video", domain
    if domain == "labeled_objects":
        return "labeled_objects", None
    raise ValueError("unknown domain: {!r}".format(domain))


def run_tag(parameters, aeclass, U, A, P, video_category=None, fps=None):
    """Deterministic run tag with short SHA-1 suffix.

    Same `parameters` dict + same U/A/P/cat/fps => same tag => safe re-run/resume.
    Use `unique_dir()` on top to append `_2, _3 ...` if the dir already exists.
    """
    blob = json.dumps(parameters, sort_keys=True, default=str).encode("utf-8")
    h    = hashlib.sha1(blob).hexdigest()[:6]
    parts = [str(aeclass), "U{}".format(U), "A{}".format(A), "P{}".format(P)]
    if video_category:
        parts.append("cat{}".format(str(video_category).replace("/", "_")))
    if fps is not None:
        parts.append("fps{}".format(fps))
    parts.append(h)
    return "_".join(parts)


def build_out_dir(domain_seg, category_seg, tag, base=None):
    """Compose ``<base>/<domain>/<category?>/<tag>``. base defaults to OUT_DIR."""
    base = base if base is not None else OUT_DIR
    parts = [base, domain_seg]
    if category_seg:
        parts.append(category_seg)
    parts.append(tag)
    return os.path.join(*parts)


def unique_dir(base):
    """Return `base`, or `base_2 / base_3 ...` if base already exists."""
    if not os.path.exists(base):
        return base
    i = 2
    while os.path.exists("{}_{}".format(base, i)):
        i += 1
    return "{}_{}".format(base, i)


def resolved_out_dir(domain, parameters, aeclass, U, A, P,
                     video_category=None, fps=None, base=None, **dom_kw):
    """Convenience: domain_category + run_tag + build_out_dir + unique_dir.

    Used by both strips.py and sh/submit.sh (via `python3 -c`).
    """
    d, c = domain_category(domain, **dom_kw)
    tag  = run_tag(parameters, aeclass, U, A, P,
                   video_category=(video_category if d == "video" else None),
                   fps=(fps if d == "video" else None))
    return unique_dir(build_out_dir(d, c, tag, base=base))
