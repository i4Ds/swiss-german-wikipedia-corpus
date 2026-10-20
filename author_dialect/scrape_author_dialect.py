"""Refresh explicit author dialect self-declarations from als.wikipedia.

The offline author pipeline uses the stub-meta-history dump for article authors.
This script only keeps the small online step that can enrich the author map:
users with the {{Dialekt (Benutzer)}} self-declaration template.

Output:
    <OUT_DIR>/author_declarations.json

Usage:
    python scrape_author_dialect.py declarations
"""
import argparse
import json
import os
import re
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_paths import ARTICLE_DUMP, HISTORY_DUMP, EXTRACTED_DIR, RESULTS_DIR

import requests


API = "https://als.wikipedia.org/w/api.php"
UA = (
    "AlsDialectResearch/0.1 "
    "(https://github.com/i4Ds/tts_data_collection_and_prep; TTS dialect research)"
)
OUT_DIR = RESULTS_DIR
DECL_FILE = OUT_DIR / "author_declarations.json"
REQUEST_DELAY = 0.6

_session = requests.Session()
_session.headers.update({"User-Agent": UA})


def api_get(params, max_retries=6):
    params = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
    delay = 1.0
    for _ in range(max_retries):
        response = _session.get(API, params=params, timeout=40)
        if response.status_code == 429:
            time.sleep(float(response.headers.get("Retry-After", delay)))
            delay = min(delay * 2, 30)
            continue
        data = response.json()
        if "error" in data and data["error"].get("code") == "maxlag":
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        return data
    raise RuntimeError("max retries exceeded for " + str(params))


def scrape_declarations():
    print("Finding users with {{Dialekt (Benutzer)}} ...")
    users, cont = [], {}
    while True:
        data = api_get({
            "action": "query",
            "list": "embeddedin",
            "eititle": "Vorlage:Dialekt (Benutzer)",
            "einamespace": "2",
            "eilimit": "500",
            **cont,
        })
        users += [page["title"] for page in data["query"]["embeddedin"]]
        if "continue" not in data:
            break
        cont = data["continue"]
        time.sleep(REQUEST_DELAY)
    print(f"  {len(users)} users carry the template.")

    tmpl_re = re.compile(r"\{\{\s*Dialekt \(Benutzer\)\s*\|([^}]*)\}\}", re.IGNORECASE)
    out = {}
    for i, title in enumerate(users):
        data = api_get({
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvprop": "content",
            "rvslots": "main",
            "rvlimit": 1,
        })
        page = data["query"]["pages"][0]
        if "revisions" not in page:
            continue
        content = page["revisions"][0]["slots"]["main"]["content"]
        match = tmpl_re.search(content)
        args = [arg.strip() for arg in match.group(1).split("|")] if match else []
        out[title.replace("Benutzer:", "")] = {
            "raw_args": args,
            "dialect_hg": args[0] if len(args) > 0 else None,
            "dialect_fine": args[1] if len(args) > 1 else None,
        }
        if (i + 1) % 10 == 0:
            print(f"  parsed {i + 1}/{len(users)}")
        time.sleep(REQUEST_DELAY)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DECL_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} declarations -> {DECL_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("declarations")
    args = parser.parse_args()

    if args.cmd == "declarations":
        scrape_declarations()
