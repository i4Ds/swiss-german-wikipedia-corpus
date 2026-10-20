"""Fetch dialect category membership from the Alemannic Wikipedia API.

Queries all subcategories of Kategorie:Wikipedia:Dialekt and builds a mapping:
    {page_id (str): dialect_name (str) | null}
where:
    str   → Swiss dialect name (e.g. "Züritüütsch")
    null  → non-Swiss article (should be dropped by extract_als_articles.py)

Articles not present in the map carry no dialect tag → caller defaults to "ch".

Output: als_dialect_map.json
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from corpus_paths import ARTICLE_DUMP, HISTORY_DUMP, EXTRACTED_DIR, RESULTS_DIR

SCRIPT_DIR = EXTRACTED_DIR
OUTPUT_FILE = SCRIPT_DIR / "als_dialect_map.json"

ALS_API = "https://als.wikipedia.org/w/api.php"
USER_AGENT = "ParallelCorpusBuilder/1.0 (Alemannic-German TTS research)"
REQUEST_DELAY = 0.3  # seconds between API calls

# ── dialect classification ────────────────────────────────────────────────────

# Categories whose articles are NOT Swiss German → mark for removal.
# Based on the full category list from Kategorie:Wikipedia:Dialekt.
NON_SWISS_CATEGORIES = {
    "Kategorie:Artikel uf Allgairisch",         # Allgäu, Germany
    "Kategorie:Artikel uf Bodeseealemannisch",  # Lake Constance, Germany/Austria
    "Kategorie:Artikel uf Dinkelbärgisch",      # Baden, Germany
    "Kategorie:Artikel uf Elsassisch",          # Alsace, France
    "Kategorie:Artikel uf Hotzewälderisch",     # Hotzenwald/Black Forest, Germany
    "Kategorie:Artikel uf Jùutütsch",           # Judeo-Alemannic
    "Kategorie:Artikel uf Liachtastänerisch",   # Liechtenstein
    "Kategorie:Artikel uf Markgräflerisch",     # Baden, Germany
    "Kategorie:Artikel uf Milhüserisch",        # Mulhouse/Alsace, France
    "Kategorie:Artikel uf Oberrhiinalemannisch",# Rhine border region (France/Germany)
    "Kategorie:Artikel uf Schwäbisch",          # Swabia, Germany
    "Kategorie:Artikel uf Strossburjerditsch",  # Strasbourg, Alsace
    "Kategorie:Artikel uf Südschwarzwälderisch",# Black Forest, Germany
    "Kategorie:Artikel uf Vorarlbärgisch",      # Vorarlberg, Austria
}

# ── API helpers ───────────────────────────────────────────────────────────────

def api_get(params: dict) -> dict:
    url = ALS_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_subcategories(parent: str) -> list[str]:
    """Return all subcategory titles under parent (single page, ≤100 expected)."""
    data = api_get({
        "action": "query",
        "list": "categorymembers",
        "cmtitle": parent,
        "cmtype": "subcat",
        "cmlimit": "100",
        "format": "json",
        "formatversion": "2",
    })
    return [p["title"] for p in data["query"]["categorymembers"]]


def fetch_category_members(cat_title: str) -> list[str]:
    """Return all page IDs (as strings) in the given category, paginated."""
    members: list[str] = []
    cont: dict = {}
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": cat_title,
            "cmtype": "page",
            "cmprop": "ids",
            "cmlimit": "500",
            "format": "json",
            "formatversion": "2",
            **cont,
        }
        data = api_get(params)
        for m in data["query"]["categorymembers"]:
            members.append(str(m["pageid"]))
        if "continue" not in data:
            break
        cont = data["continue"]
        time.sleep(REQUEST_DELAY)
    return members


def dialect_label(cat_title: str) -> str:
    """Strip 'Kategorie:Artikel uf ' prefix to get the dialect name."""
    prefix = "Kategorie:Artikel uf "
    return cat_title[len(prefix):] if cat_title.startswith(prefix) else cat_title.split(":")[-1]


# ── main ──────────────────────────────────────────────────────────────────────

def build_dialect_map() -> None:
    print("Fetching dialect subcategories from als.wikipedia.org …")
    subcats = fetch_subcategories("Kategorie:Wikipedia:Dialekt")
    print(f"  Found {len(subcats)} dialect categories.\n")

    # {page_id: dialect_name_or_None}
    # None  = non-Swiss (drop)
    # str   = Swiss dialect label
    dialect_map: dict[str, str | None] = {}
    conflicts = 0

    for cat in sorted(subcats):
        is_swiss = cat not in NON_SWISS_CATEGORIES
        dialect: str | None = dialect_label(cat) if is_swiss else None
        label = dialect if dialect else f"[non-Swiss: {dialect_label(cat)}]"

        members = fetch_category_members(cat)
        print(f"  {len(members):5d}  {label}")

        for pid in members:
            if pid in dialect_map:
                existing = dialect_map[pid]
                # Prefer a real Swiss dialect name over None or another name
                if existing is None and dialect is not None:
                    dialect_map[pid] = dialect
                conflicts += 1
            else:
                dialect_map[pid] = dialect

        time.sleep(REQUEST_DELAY)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dialect_map, f, ensure_ascii=False)

    swiss_count = sum(1 for v in dialect_map.values() if v is not None)
    non_swiss_count = sum(1 for v in dialect_map.values() if v is None)
    print(f"\nDone.")
    print(f"  {swiss_count:,} Swiss-tagged page IDs")
    print(f"  {non_swiss_count:,} non-Swiss page IDs (will be dropped)")
    if conflicts:
        print(f"  {conflicts} pages appeared in multiple dialect categories.")
    print(f"Output → {OUTPUT_FILE.name}")


if __name__ == "__main__":
    build_dialect_map()
