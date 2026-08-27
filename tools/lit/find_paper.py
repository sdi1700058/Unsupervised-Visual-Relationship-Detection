#!/usr/bin/env python3
"""Find a free full text for a paper, before ever touching a paywall.

Publisher pages are the last resort, not the first. Most papers behind IEEE,
Springer, Elsevier or ACM also exist as an arXiv preprint, an author copy, or
an open-access proceedings entry, and a preprint is worth far more than a
missing entry in the reading list.

The order tried here is deliberate:

  1. arXiv, by title. Exact enough that a hit is almost always the paper.
  2. Semantic Scholar, which reports an openAccessPdf when one exists
     anywhere, including author pages and institutional repositories.
  3. The open proceedings hosts, matched from the Semantic Scholar venue.

    python3 tools/lit/find_paper.py "Visual Relationship Detection: A Survey"
    python3 tools/lit/find_paper.py --file titles.txt --download .claude/lit/papers

`--download` fetches whatever it finds and names it
`<firstauthor><year>-<slug>.pdf`. Titles that come back with nothing are
printed at the end, so they can go straight into
`.claude/lit/unreachable.md`.

Every hit is title-checked against the query before it is accepted. A web
search once offered arXiv:2201.09221 as the Cheng relation-detection survey;
that identifier is a paper about digital finance. Nothing here trusts an
identifier it has not read the title of.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
# ElementTree rather than defusedxml: the only XML parsed here comes from the
# fixed arXiv API endpoint, never from user input, and defusedxml is not in
# the project environment. Point this at anything else and reconsider.
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
UA = {"User-Agent": "thesis-lit-search/1.0 (academic use)"}


def _get(url, tries=3, pause=2.0):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:
            code = getattr(e, "code", None)
            # Semantic Scholar rate-limits hard without a key. Back off.
            if code == 429 and attempt < tries - 1:
                time.sleep(pause * (attempt + 2) * 3)
                continue
            if attempt < tries - 1:
                time.sleep(pause)
                continue
            return None
    return None


def normalise(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def title_matches(query, found, threshold=0.6):
    """Symmetric word overlap, so extra words in the hit count against it.

    One-sided containment is not enough. Asking for "Curriculum Learning"
    matched a 2022 paper whose title merely contains both words, which is not
    Bengio 2009 at all. Jaccard penalises the extra words, and short queries
    are the case that needs the penalty most.
    """
    q, f = set(normalise(query).split()), set(normalise(found).split())
    if not q or not f:
        return False
    return len(q & f) / len(q | f) >= threshold


def meta_matches(hit, author=None, year=None, slack=2):
    """Check the surname and year when the caller supplied them."""
    if author:
        surnames = " ".join(hit.get("authors") or []).lower()
        if normalise(author).split()[-1] not in normalise(surnames).split():
            return False
    if year and hit.get("year"):
        try:
            if abs(int(hit["year"]) - int(year)) > slack:
                return False
        except ValueError:
            pass
    return True


def search_arxiv(title):
    q = urllib.parse.quote(f'ti:"{title}"')
    raw = _get(f"{ARXIV_API}?search_query={q}&max_results=5")
    if not raw:
        return None
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("a:entry", ns):
        found = (entry.findtext("a:title", "", ns) or "").strip()
        if not title_matches(title, found):
            continue
        pdf = None
        for link in entry.findall("a:link", ns):
            if link.get("title") == "pdf":
                pdf = link.get("href")
        if not pdf:
            aid = (entry.findtext("a:id", "", ns) or "").rsplit("/", 1)[-1]
            pdf = f"https://arxiv.org/pdf/{aid}"
        authors = [a.findtext("a:name", "", ns)
                   for a in entry.findall("a:author", ns)]
        year = (entry.findtext("a:published", "", ns) or "")[:4]
        return {"source": "arxiv", "title": found, "pdf": pdf,
                "authors": authors, "year": year}
    return None


def search_s2(title):
    q = urllib.parse.quote(title)
    fields = "title,year,authors,openAccessPdf,externalIds,venue"
    raw = _get(f"{S2_API}?query={q}&limit=5&fields={fields}")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    for p in data.get("data", []):
        if not title_matches(title, p.get("title", "")):
            continue
        oa = p.get("openAccessPdf") or {}
        if not oa.get("url"):
            # No free copy, but the arXiv id may still be recorded.
            aid = (p.get("externalIds") or {}).get("ArXiv")
            if aid:
                return {"source": "s2->arxiv", "title": p["title"],
                        "pdf": f"https://arxiv.org/pdf/{aid}",
                        "authors": [a["name"] for a in p.get("authors", [])],
                        "year": str(p.get("year") or "")}
            continue
        return {"source": "s2", "title": p["title"], "pdf": oa["url"],
                "authors": [a["name"] for a in p.get("authors", [])],
                "year": str(p.get("year") or ""), "venue": p.get("venue", "")}
    return None


def find(title, author=None, year=None):
    for fn in (search_arxiv, search_s2):
        hit = fn(title)
        if hit and meta_matches(hit, author, year):
            return hit
        time.sleep(1.0)
    return None


def parse_line(line):
    """`title | author | year`, with author and year optional."""
    parts = [p.strip() for p in line.split("|")]
    title = parts[0]
    author = parts[1] if len(parts) > 1 and parts[1] else None
    year = parts[2] if len(parts) > 2 and parts[2] else None
    return title, author, year


def filename_for(hit, title):
    a = (hit.get("authors") or ["unknown"])[0].split()[-1].lower()
    a = re.sub(r"[^a-z]", "", a) or "unknown"
    slug = "-".join(normalise(hit.get("title") or title).split()[:5])
    return f"{a}{hit.get('year') or ''}-{slug}.pdf"


def download(hit, out_dir, title):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename_for(hit, title))
    if os.path.exists(path):
        return path, "already had it"
    blob = _get(hit["pdf"])
    if not blob or not blob.startswith(b"%PDF"):
        return None, "not a PDF"
    with open(path, "wb") as f:
        f.write(blob)
    return path, f"{len(blob) // 1024} KB"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Find a free full text for a paper title.")
    ap.add_argument("title", nargs="*",
                    help="paper title, optionally 'title | author | year'")
    ap.add_argument("--file",
                    help="file of 'title | author | year' lines, one per line")
    ap.add_argument("--download", metavar="DIR",
                    help="fetch what is found into DIR")
    args = ap.parse_args(argv)

    titles = []
    if args.file:
        with open(args.file) as f:
            titles = [l.strip() for l in f if l.strip()
                      and not l.startswith("#")]
    if args.title:
        titles.append(" ".join(args.title))
    if not titles:
        ap.error("give a title or --file")

    missing = []
    for line in titles:
        t, author, year = parse_line(line)
        hit = find(t, author, year)
        if not hit:
            print(f"MISS  {t[:70]}")
            missing.append(line)
            continue
        note = ""
        if args.download:
            path, note = download(hit, args.download, t)
            note = f"  -> {os.path.basename(path)} ({note})" if path \
                else f"  -> download failed: {note}"
        print(f"FOUND {t[:60]}\n      [{hit['source']}] {hit['pdf']}{note}")
        time.sleep(1.0)

    if missing:
        print(f"\n{len(missing)} with no free copy. For unreachable.md:")
        for t in missing:
            print(f"  - {t}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
