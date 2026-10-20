"""Extract per-article author signals OFFLINE from the stub-meta-history dump.

No internet required -> runs anywhere (SLURM CPU node, login node, laptop).
This replaces the API creator-scrape: the dump already lists every revision's
contributor + byte size, so we can derive author signals for ALL articles at once.

For each main-namespace (ns=0) page we emit, per article:
    id, title, num_revisions, creation_ts,
    creator              (author of the oldest revision),
    creator_anon         (oldest revision was an anonymous IP edit),
    top_author_edits     (contributor with the most revisions),
    top_author_bytes     (contributor with the most NET bytes added; ~content author),
    category_dialect     (existing category tag from als_dialect_map.json, or null)

Output: <OUT_DIR>/article_authors_from_dump.jsonl

Usage:
    python parse_creators_from_dump.py                 # full dump
    python parse_creators_from_dump.py --limit 2000    # subsample (try-around)
"""
import argparse
import gzip
import json
import os
import re
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_paths import ARTICLE_DUMP, HISTORY_DUMP, EXTRACTED_DIR, RESULTS_DIR
import xml.etree.ElementTree as ET

DUMP = HISTORY_DUMP
DIALECT_MAP = EXTRACTED_DIR / "als_dialect_map.json"
OUT_DIR = RESULTS_DIR
OUT_FILE = OUT_DIR / "article_authors_from_dump.jsonl"


def local(tag: str) -> str:
    """Strip the {namespace} prefix from an ElementTree tag."""
    return tag.rsplit("}", 1)[-1]


def _is_ip(s: str) -> bool:
    """True for raw IPv4/IPv6 usernames (anonymous edits)."""
    return bool(s) and all(c in "0123456789.:abcdefABCDEF" for c in s) and (
        s.count(".") == 3 or ":" in s)


def parse(limit: int = 0):
    dialect_map = json.load(open(DIALECT_MAP, encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_pages = n_articles = 0
    with gzip.open(DUMP, "rb") as f, open(OUT_FILE, "w", encoding="utf-8") as out:
        for _, elem in ET.iterparse(f, events=("end",)):
            if local(elem.tag) != "page":
                continue
            n_pages += 1

            ns = elem.findtext("{*}ns")
            if ns != "0":
                elem.clear()
                continue

            page_id = elem.findtext("{*}id")
            title = elem.findtext("{*}title")

            # Walk revisions in document order (oldest first).
            creator = creator_anon = creation_ts = None
            edits = defaultdict(int)      # contributor -> #revisions
            net_bytes = defaultdict(int)  # contributor -> net bytes added
            prev_bytes = 0
            n_rev = 0
            for rev in elem.iterfind("{*}revision"):
                contrib = rev.find("{*}contributor")
                if contrib is not None and contrib.find("{*}username") is not None:
                    user = contrib.findtext("{*}username")
                    anon = False
                elif contrib is not None and contrib.find("{*}ip") is not None:
                    user = contrib.findtext("{*}ip")
                    anon = True
                else:
                    user, anon = None, True  # deleted/suppressed contributor

                ts = rev.findtext("{*}timestamp")
                text_el = rev.find("{*}text")
                cur_bytes = int(text_el.get("bytes", 0)) if text_el is not None else 0

                if n_rev == 0:
                    creator, creator_anon, creation_ts = user, anon, ts
                if user is not None:
                    edits[user] += 1
                    delta = cur_bytes - prev_bytes
                    if delta > 0:
                        net_bytes[user] += delta
                prev_bytes = cur_bytes
                n_rev += 1

            top_edits = max(edits, key=edits.get) if edits else None
            top_bytes = max(net_bytes, key=net_bytes.get) if net_bytes else None

            # Registered (non-IP) contributors ranked by net bytes added, so a
            # downstream step can skip maintainers/bots and fall back to the next
            # human author whose dialect is actually known. Capped to keep it small.
            ranked = sorted(
                ((u, b) for u, b in net_bytes.items() if not _is_ip(u)),
                key=lambda kv: kv[1], reverse=True)[:15]

            rec = {
                "id": page_id,
                "title": title,
                "num_revisions": n_rev,
                "creation_ts": creation_ts,
                "creator": creator,
                "creator_anon": creator_anon,
                "top_author_edits": top_edits,
                "top_author_bytes": top_bytes,
                "authors_by_bytes": ranked,
                "category_dialect": dialect_map.get(page_id),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_articles += 1
            elem.clear()

            if n_articles % 5000 == 0:
                print(f"  {n_articles:,} articles ({n_pages:,} pages seen)")
            if limit and n_articles >= limit:
                break

    print(f"Done. {n_articles:,} ns=0 articles -> {OUT_FILE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = full dump")
    parse(ap.parse_args().limit)
