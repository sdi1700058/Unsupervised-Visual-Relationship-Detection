#!/usr/bin/env bash
#
# unpack_havid_annotations.sbatch — HA-ViD's annotations only, no images
#
#   sbatch --job-name=havid-ann runs/data/unpack_havid_annotations.sbatch
#
# WHY. ObjectDetection.zip is 117,591,123,942 bytes and holds two STORED
# zips, data/train.zip and data/val.zip, which hold PNG frames and a JSON
# annotation file each. Stored members can be opened in place, so this reads
# the inner central directories and extracts only what is not an image. From
# ActionSegmentation.zip it takes mapping.txt, groundTruth and splits, not the
# I3D features. The temporal annotation zip is small and taken whole.
#
# Everything lands under data/video/ha_vid/ext/. The frames stay packed.
#
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load python/3.9.0
python3 - "${ROOT}/data/video/ha_vid/raw" "${ROOT}/data/video/ha_vid/ext" <<'PY'
import collections, os, sys, zipfile
raw, ext = sys.argv[1], sys.argv[2]
IMAGES = (".png", ".jpg", ".jpeg")

def take(z, info, dest):
    path = os.path.join(dest, info.filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with z.open(info) as src, open(path, "wb") as out:
        while True:
            block = src.read(1 << 20)
            if not block:
                break
            out.write(block)
    return path

outer = zipfile.ZipFile(os.path.join(raw, "ObjectDetection.zip"))
for meta in ("data/readme.txt", "data/ObjectDetection_data_video_info.csv"):
    take(outer, outer.getinfo(meta), os.path.join(ext, "ObjectDetection"))
for split in ("train", "val"):
    inner = zipfile.ZipFile(outer.open("data/%s.zip" % split))
    infos = [i for i in inner.infolist() if not i.is_dir()]
    kinds = collections.Counter(os.path.splitext(i.filename)[1].lower()
                                for i in infos)
    print("%s.zip: %d files, by type %s" % (split, len(infos), dict(kinds)),
          flush=True)
    for i in infos:
        if not i.filename.lower().endswith(IMAGES):
            p = take(inner, i, os.path.join(ext, "ObjectDetection", split))
            print("  took %s, %d bytes" % (p, i.file_size), flush=True)

seg = zipfile.ZipFile(os.path.join(raw, "ActionSegmentation.zip"))
n = 0
for i in seg.infolist():
    name = i.filename
    if i.is_dir() or name.startswith("data/features/"):
        continue
    take(seg, i, os.path.join(ext, "ActionSegmentation"))
    n += 1
print("ActionSegmentation: took %d files" % n)

tem = zipfile.ZipFile(os.path.join(raw, "HAViD_temporalAnnotation.zip"))
tem.extractall(ext)
print("temporal annotation: %d members" % len(tem.namelist()))
PY
du -sh "${ROOT}/data/video/ha_vid/ext"
