#!/usr/bin/env python3
"""Build a machine-readable index of papers, the datasets they use, and their metrics.

**Why this is a file and not a set of links written by hand.** The reference
notes are a library of disconnected facts: 118 paper summaries, a datasets
document and an evaluation document that never point at each other. The question
worth answering — *which datasets and metrics do the papers nearest this thesis
actually use* — cannot be read off them, and the supervisors asked it directly:
*"Ποια βρήκες; Τι datasets χρησιμοποιούσαν;"*

Hand-maintained cross-references would rot exactly the way the numbers rotted,
three times. So the edges are extracted from the summaries, stored once, and the
prose is generated from them.

**Extraction is partial and says so.** Each summary carries a `Data.` bullet and
an `Evaluation.` bullet in free prose. Dataset and metric names are matched
against curated vocabularies below. Anything unmatched is **reported, never
dropped**, because a coverage figure that looks complete is worse than one that
admits a gap.

    python3 tools/lit/build_index.py                # build and report coverage
    python3 tools/lit/build_index.py --unmatched     # show what failed to match

Writes `notes/lit/index.json`. Standard library only, Python 3.6 clean.
"""

import argparse
import collections
import json
import os
import re


SOURCE = "notes/docs/RELATED_WORK.md"
OUT = "notes/lit/index.json"

# Canonical dataset name -> the spellings that appear in the summaries. Curated
# by reading the frequent proper nouns out of the Data bullets, not guessed.
DATASETS = {
    "VidVRD": ["vidvrd", "imagenet-vidvrd", "imagenet vidvrd"],
    "VidOR": ["vidor"],
    "ImageNet-VID": ["imagenet-vid", "imagenet vid", "ilsvrc2015 vid"],
    "Action Genome": ["action genome"],
    "Charades": ["charades"],
    "Visual Genome": ["visual genome"],
    "Something-Something": ["something-something", "something something"],
    "Something-Else": ["something-else"],
    "Kinetics": ["kinetics"],
    "HACS": ["hacs"],
    "Human3.6M": ["human3.6m", "human3.6", "h3.6m"],
    "CMU Mocap": ["cmu mocap", "cmu motion capture"],
    "AMASS": ["amass"],
    "CLEVRER": ["clevrer"],
    "CATER": ["cater"],
    "CLEVR": ["clevr"],
    "MNIST": ["mnist"],
    "Blocksworld": ["blocksworld", "blocks world"],
    "OpenImages": ["openimages", "open images"],
    "Flickr30K": ["flickr30k"],
    "GQA": ["gqa"],
    "COCO": ["coco", "ms-coco"],
    "VRD": ["vrd dataset", "visual relationship detection dataset"],
    "BridgeData": ["bridgedata", "bridge data"],
    "RT-1": ["rt-1"],
    "Ego4D": ["ego4d"],
    "EPIC-KITCHENS": ["epic-kitchens", "epic kitchens"],
    "METR-LA": ["metr-la"],
    "YFCC100M": ["yfcc100m"],
    "Sokoban": ["sokoban"],
    "LightsOut": ["lightsout", "lights out"],
    "15-Puzzle": ["15-puzzle", "15 puzzle"],
    "8-puzzle": ["8-puzzle", "8 puzzle"],
}

# Canonical metric -> spellings. Deliberately small: only what actually appears.
METRICS = {
    "mAP": ["map", "mean average precision"],
    "Recall@K": ["r@", "recall@", "recall at"],
    "accuracy": ["accuracy", "top-1", "top-5"],
    "success rate": ["success rate"],
    "MPJPE": ["mpjpe", "mean per joint position error"],
    "IoU": ["iou", "viou", "intersection over union"],
    "MSE": ["mse", "mean squared error"],
    "RMSE": ["rmse"],
    "AUC": ["auc"],
    "F1": ["f1"],
    "plan length": ["plan length", "plan cost", "solution length"],
    "coverage": ["coverage"],
    "human evaluation": ["human evaluation", "human study"],
}

# Whether the data is of the real world. This is the thesis's own argument, so
# it is a first-class field rather than something to read out of prose later.
REALNESS = {
    "real": ["real video", "real world", "real-world", "real images",
             "real robot", "real interaction", "real data", "real."],
    "synthetic": ["synthetic", "rendered", "procedurally generated"],
    "simulated": ["simulated", "simulation", "in simulation"],
}

# Summaries state the negative explicitly and often: "All rendered. No real
# video." Matching "real video" there records the opposite of what it says, on
# the one field that carries the thesis's own argument. So negated spans are
# removed before the realness match runs.
NEGATED = re.compile(r"\b(?:no|not|never|rather than|instead of|nor)\s+"
                     r"(?:a\s+|any\s+)?(?:real|synthetic|simulated)"
                     r"[a-z-]*(?:\s+\w+)?", re.I)

BULLETS = ("Problem", "How", "Data", "Evaluation", "Performance")


# The summaries use two list styles for the five points, a dash and a number.
# Matching only the dash silently lost 54 of 118 Data bullets.
_ITEM = r"(?:-|\d+\.)\s+"


def _bullet(entry, name):
    """One five-point bullet's text, flattened, or an empty string."""
    found = re.search(r"%s\*\*%s\.\*\*(.*?)(?=\n%s\*\*|\n\n|\Z)"
                      % (_ITEM, name, _ITEM), entry, re.S)
    return re.sub(r"\s+", " ", found.group(1)).strip() if found else ""


def _match(text, vocabulary):
    """Canonical names whose spellings appear in `text`, on word boundaries.

    Substring matching is wrong here: "CLEVR" would match every mention of
    CLEVRER, and "mAP" would match "mapping".
    """
    low = text.lower()
    out = []
    for name, spellings in vocabulary.items():
        for spelling in spellings:
            if re.search(r"(?<![a-z0-9])%s(?![a-z0-9])"
                         % re.escape(spelling), low):
                out.append(name)
                break
    return sorted(out)


def split_entries(text):
    """The paper summaries, each starting at its `### <id>.` heading."""
    return [e for e in re.split(r"\n(?=### [A-Z]\d+\.)", text)
            if re.match(r"### [A-Z]\d+\.", e)]


def parse_entry(entry):
    """One paper, as a record."""
    head = entry.split("\n", 1)[0]
    found = re.match(r"### ([A-Z]\d+)\.\s*(.*)", head)
    paper_id, rest = found.group(1), found.group(2)
    url = ""
    link = re.search(r"\[([^\]]+)\]\((https?://[^)]+)\)", rest)
    if link:
        url = link.group(2)
    title = re.split(r"\s+[—·]\s+", rest)[0].strip()

    data_text = _bullet(entry, "Data")
    eval_text = _bullet(entry, "Evaluation")
    affirmative = NEGATED.sub(" ", data_text)
    realness = [k for k, spellings in REALNESS.items()
                if any(s in affirmative.lower() for s in spellings)]
    return {
        "id": paper_id,
        "title": re.sub(r"\[|\]\([^)]*\)", "", title).strip(),
        "url": url,
        "datasets": _match(data_text, DATASETS),
        "metrics": _match(eval_text, METRICS),
        "realness": realness,
        "data_text": data_text,
        "eval_text": eval_text,
        "has_mechanism": "**Mechanism.**" in entry,
    }


def shortlist_ids(text):
    """The ranked shortlist, in order, from its table."""
    section = text.split("## The shortlist", 1)
    if len(section) < 2:
        return []
    table = section[1].split("\n## ", 1)[0]
    ids = []
    for row in re.findall(r"^\|\s*(\d+|—)\s*\|\s*\*\*([A-Z]\d+)\.",
                          table, re.M):
        ids.append(row[1])
    return ids


def build(source=None):
    source = source or SOURCE
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    papers = [parse_entry(e) for e in split_entries(text)]
    ranked = shortlist_ids(text)
    by_id = dict((p["id"], p) for p in papers)
    for rank, pid in enumerate(ranked, 1):
        if pid in by_id:
            by_id[pid]["shortlist_rank"] = rank

    dataset_use = collections.Counter()
    metric_use = collections.Counter()
    for p in papers:
        for d in p["datasets"]:
            dataset_use[d] += 1
        for m in p["metrics"]:
            metric_use[m] += 1

    return {
        "source": source,
        "papers": papers,
        "shortlist": ranked,
        "dataset_use": dict(dataset_use),
        "metric_use": dict(metric_use),
        "coverage": {
            "papers": len(papers),
            "with_data_bullet": sum(1 for p in papers if p["data_text"]),
            "with_eval_bullet": sum(1 for p in papers if p["eval_text"]),
            "with_a_dataset_matched": sum(1 for p in papers if p["datasets"]),
            "with_a_metric_matched": sum(1 for p in papers if p["metrics"]),
            "with_realness": sum(1 for p in papers if p["realness"]),
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--unmatched", action="store_true",
                    help="print entries whose Data or Evaluation matched nothing")
    ap.add_argument("--source", default=None)
    a = ap.parse_args(argv)

    index = build(a.source)
    c = index["coverage"]
    print("papers %d; data bullets %d; evaluation bullets %d"
          % (c["papers"], c["with_data_bullet"], c["with_eval_bullet"]))
    print("matched a dataset: %d (%.0f%%)   a metric: %d (%.0f%%)   "
          "real/synthetic: %d (%.0f%%)"
          % (c["with_a_dataset_matched"],
             100.0 * c["with_a_dataset_matched"] / c["papers"],
             c["with_a_metric_matched"],
             100.0 * c["with_a_metric_matched"] / c["papers"],
             c["with_realness"], 100.0 * c["with_realness"] / c["papers"]))
    print()
    print("most used datasets:")
    for name, n in sorted(index["dataset_use"].items(), key=lambda kv: -kv[1])[:10]:
        print("  %-22s %d" % (name, n))
    print("most used metrics:")
    for name, n in sorted(index["metric_use"].items(), key=lambda kv: -kv[1])[:10]:
        print("  %-22s %d" % (name, n))

    if a.unmatched:
        print("\nno dataset matched, Data bullet shown verbatim:")
        for p in index["papers"]:
            if not p["datasets"] and p["data_text"]:
                print("  %-5s %s" % (p["id"], p["data_text"][:96]))

    directory = os.path.dirname(OUT)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(OUT, "w") as handle:
        json.dump(index, handle, indent=2)
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
