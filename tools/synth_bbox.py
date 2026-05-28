#!/usr/bin/env python3
"""tools/synth_bbox.py — ONE-TIME bbox preprocessing for VideoNet hand-skill clips.

Pipeline (per frame):
  MediaPipe.Hands  → up to 2 hands × 21 landmarks
  Variant A: 1 bbox/hand (aggregate over all landmarks)         → 2 slots
  Variant B: 5 bboxes/hand (one per finger landmark group)      → 10 slots
  Grounded-DINO (optional, text prompt e.g. "pen", "rope")      → +1 held-object slot

Output: VidVRD-schema JSON with `trajectories` indexed by source-PTS frame
index (matches puzzle_vidvrd.build_dataset's `f"{fid:06d}.jpg"` lookup).

Run ONCE per video in the sidecar venv-detect. JSONs persist on disk;
never re-run at training time.

Usage (inside venv-detect):
    source venv-detect/bin/activate
    python tools/synth_bbox.py \
        --frames-dir data/video/videonet/frames_5fps/<UUID> \
        --video-id   <UUID> \
        --variant    A \
        --object-prompt pen \
        --out        data/video/videonet/annotations/A/train/<UUID>.json
    python tools/synth_bbox.py --variant B --out .../B/train/<UUID>.json ...

Env knobs
    HF_GDINO_MODEL  default IDEA-Research/grounding-dino-tiny
    HF_GDINO_THRESH default 0.25 (box score threshold)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Pin HF cache inside the workspace BEFORE any HF import so the G-DINO
# checkpoint downloads under labeled-fosae/.hf instead of $HOME/.cache.
_PROJECT_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(_PROJECT_DIR / ".hf"))
os.environ.setdefault("HF_HUB_CACHE", str(_PROJECT_DIR / ".hf" / "hub"))

import numpy as np
from PIL import Image

try:
    import mediapipe as mp
except ImportError:
    sys.exit("mediapipe missing. Run: bash sh/setup_venv_detect.sh && source venv-detect/bin/activate")


# MediaPipe.Hands landmark groups (21 lm/hand; wrist=0; 4 lm/finger)
_FINGER_GROUPS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}
_FINGERS_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def _bbox_from_landmarks(lms, idx_subset, W, H, margin=4):
    """Axis-aligned bbox over a subset of normalized landmarks, clipped to image."""
    xs = [lms[i].x for i in idx_subset]
    ys = [lms[i].y for i in idx_subset]
    xmin = int(max(0, min(xs) * W - margin))
    ymin = int(max(0, min(ys) * H - margin))
    xmax = int(min(W - 1, max(xs) * W + margin))
    ymax = int(min(H - 1, max(ys) * H + margin))
    if xmax <= xmin or ymax <= ymin:
        return None
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def _detect_hands(frame_np, hands):
    """Return list of (label, landmarks). label = 'Left' or 'Right' from MP."""
    res = hands.process(frame_np)
    if not res.multi_hand_landmarks:
        return []
    out = []
    handedness = res.multi_handedness or []
    for i, lms in enumerate(res.multi_hand_landmarks):
        label = "Right"
        if i < len(handedness) and handedness[i].classification:
            label = handedness[i].classification[0].label
        out.append((label, lms.landmark))
    return out


def _variant_a_entries(detected, W, H):
    """Return list of {'tid': 0|1, 'bbox': ...} for one frame (left=0, right=1)."""
    out = []
    seen_labels = set()
    for label, lms in detected:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        bbox = _bbox_from_landmarks(lms, list(range(21)), W, H, margin=8)
        if bbox is None:
            continue
        tid = 0 if label == "Left" else 1
        out.append({"tid": tid, "bbox": bbox})
    return out


def _variant_b_entries(detected, W, H):
    """5 finger bboxes/hand. tids: left 0-4 (thumb..pinky), right 5-9."""
    out = []
    seen_labels = set()
    for label, lms in detected:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        base = 0 if label == "Left" else 5
        for fi, fname in enumerate(_FINGERS_ORDER):
            bbox = _bbox_from_landmarks(lms, _FINGER_GROUPS[fname], W, H, margin=4)
            if bbox is None:
                continue
            out.append({"tid": base + fi, "bbox": bbox})
    return out


def _make_object_categories(variant):
    if variant == "A":
        return [{"tid": 0, "category": "left_hand"},
                {"tid": 1, "category": "right_hand"}]
    cats = []
    for hand_label, base in (("l", 0), ("r", 5)):
        for fi, fname in enumerate(_FINGERS_ORDER):
            cats.append({"tid": base + fi, "category": f"{hand_label}_{fname}"})
    return cats


def _held_obj_tid(variant):
    return 2 if variant == "A" else 10


def _load_gdino(prompt):
    """Lazy-load Grounded-DINO via HF transformers. None if --object-prompt empty."""
    if not prompt:
        return None
    from transformers import pipeline
    model_id = os.environ.get("HF_GDINO_MODEL", "IDEA-Research/grounding-dino-tiny")
    print(f"[gdino] loading {model_id} (one-time download ~1.5 GB)")
    return pipeline("zero-shot-object-detection", model=model_id)


def _detect_obj(detector, frame_pil, prompt, thresh):
    """Top-1 bbox for `prompt`. Returns dict or None."""
    if detector is None:
        return None
    # transformers pipeline expects candidate_labels list w/o trailing period
    res = detector(frame_pil,
                   candidate_labels=[prompt],
                   threshold=thresh)
    if not res:
        return None
    # res is list of dicts: {'score','label','box':{xmin,ymin,xmax,ymax}}
    best = max(res, key=lambda r: r["score"])
    b = best["box"]
    return {"xmin": int(b["xmin"]), "ymin": int(b["ymin"]),
            "xmax": int(b["xmax"]), "ymax": int(b["ymax"])}


def synth(frames_dir, video_id, variant, object_prompt, out_path,
          mp_complexity=1, gdino_thresh=0.25):
    frame_paths = sorted(Path(frames_dir).glob("*.jpg"))
    if not frame_paths:
        sys.exit(f"no .jpg frames in {frames_dir}")

    # Map source-PTS index → frame_path. Filename = "<pts>.jpg".
    pts_to_path = {}
    for fp in frame_paths:
        m = re.match(r"^(\d+)$", fp.stem)
        if not m:
            print(f"[warn] skipping non-numeric frame name {fp.name}")
            continue
        pts_to_path[int(m.group(1))] = fp
    if not pts_to_path:
        sys.exit("no numeric-named frames found (expected NNNNNN.jpg)")

    # Probe image size from first frame
    w0, h0 = Image.open(next(iter(pts_to_path.values()))).size

    # Init MediaPipe.Hands (image mode, 2 hands max)
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=mp_complexity,
        min_detection_confidence=0.3)

    gdino = _load_gdino(object_prompt)
    held_tid = _held_obj_tid(variant)
    cats = _make_object_categories(variant)
    if object_prompt:
        cats.append({"tid": held_tid, "category": object_prompt})

    max_pts = max(pts_to_path)
    trajectories = [[] for _ in range(max_pts + 1)]

    processed = 0
    n_hand_frames = 0
    n_obj_frames = 0
    for pts in sorted(pts_to_path):
        fp = pts_to_path[pts]
        pil = Image.open(fp).convert("RGB")
        arr = np.array(pil)
        H, W = arr.shape[:2]

        detected = _detect_hands(arr, hands)
        if variant == "A":
            entries = _variant_a_entries(detected, W, H)
        else:
            entries = _variant_b_entries(detected, W, H)
        if entries:
            n_hand_frames += 1

        if object_prompt:
            obj_bbox = _detect_obj(gdino, pil, object_prompt, gdino_thresh)
            if obj_bbox is not None:
                entries.append({"tid": held_tid, "bbox": obj_bbox})
                n_obj_frames += 1

        trajectories[pts] = entries
        processed += 1
        if processed % 50 == 0:
            print(f"  [{processed}/{len(pts_to_path)}] pts={pts} "
                  f"hands_so_far={n_hand_frames} obj_so_far={n_obj_frames}")

    hands.close()

    ann = {
        "video_id": video_id,
        "width":    w0,
        "height":   h0,
        "subject/objects": cats,
        "trajectories":    trajectories,
        "_meta": {
            "variant":            variant,
            "object_prompt":      object_prompt or None,
            "n_frames_processed": processed,
            "n_frames_with_hands": n_hand_frames,
            "n_frames_with_object": n_obj_frames,
            "synth_tool":         "synth_bbox.py",
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ann, f)
    print(f"[ok] wrote {out_path}  states={processed}  hands={n_hand_frames}  obj={n_obj_frames}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir",    required=True)
    ap.add_argument("--video-id",      required=True)
    ap.add_argument("--variant",       choices=["A", "B"], required=True,
                    help="A=whole-hand (2 slots), B=per-finger (10 slots)")
    ap.add_argument("--object-prompt", default="",
                    help="Grounded-DINO text prompt (e.g. 'pen', 'rope'). "
                         "Empty = skip object detection (hand-only).")
    ap.add_argument("--out",           required=True, help="output JSON path")
    ap.add_argument("--mp-complexity", type=int, default=1, choices=[0, 1],
                    help="MediaPipe model complexity (0=faster, 1=more accurate)")
    ap.add_argument("--gdino-thresh",  type=float, default=0.25)
    args = ap.parse_args()
    synth(args.frames_dir, args.video_id, args.variant, args.object_prompt,
          args.out, mp_complexity=args.mp_complexity, gdino_thresh=args.gdino_thresh)


if __name__ == "__main__":
    main()
