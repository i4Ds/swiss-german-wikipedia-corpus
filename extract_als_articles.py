"""Extract and clean articles from the Alemannic Wikipedia XML dump.

Only Swiss German articles are kept.  Dialect tagging uses the pre-built
category map produced by fetch_dialect_categories.py (als_dialect_map.json):
  - page_id present, value is a string → use that Swiss dialect label
  - page_id present, value is null    → non-Swiss article, drop it
  - page_id absent                    → no dialect tag, keep with dialect="ch"

Input:  alswiki-latest-pages-articles.xml.bz2  (in same directory)
        als_dialect_map.json             (produced by fetch_dialect_categories.py)
Output: als_articles.jsonl  — one JSON object per line:
        {"id": "<page_id>", "title": "<title>", "dialect": "<dialect>", "text": "<plain_text>"}

Dependencies (optional but recommended for cleaner output):
    pip install mwparserfromhell
"""

import bz2
import json
import re
import sys
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from corpus_paths import ARTICLE_DUMP, HISTORY_DUMP, EXTRACTED_DIR, RESULTS_DIR
import xml.etree.ElementTree as ET

SCRIPT_DIR = EXTRACTED_DIR
DUMP_FILE = ARTICLE_DUMP
DIALECT_MAP_FILE = SCRIPT_DIR / "als_dialect_map.json"
OUTPUT_FILE = SCRIPT_DIR / "als_articles.jsonl"

# Maps Alemannic dialect name → Swiss canton code (ISO 3166-2:CH suffix)
DIALECT_TO_CANTON: dict[str, str] = {
    "Baseldytsch":      "bs",
    "Züritüütsch":      "zh",
    "Bärndütsch":       "be",
    "Ostschwizertütsch":"sg",
    "Wallisertiitsch":  "vs",
    "Aargauerdüütsch":  "ag",
    "Walsertiitsch":    "ch",
    "Glarnertüütsch":   "ch",
    "Schwyzerdütsch":   "ch",
    "Seislertütsch":    "ch",
    "Soledurnerdütsch": "ch",
    "Baselbieterdütsch":"bs",
    "Lozäärntüütsch":   "lu",
    "Freiämtertütsch":  "ag",
    "Innerschwyzerisch":"lu",
    "Schwizerdütsch":   "ch",
    "Pündnertütsch":    "gr",
    "St. Gallerdütsch": "sg",
}

MW_NS = "http://www.mediawiki.org/xml/export-0.11/"

# Alemannic Wikipedia namespace prefixes that are NOT main articles
SKIP_PREFIXES = (
    "Diskussion:", "Benutzer:", "Wikipedia:", "Datei:", "MediaWiki:",
    "Vorlage:", "Hilfe:", "Kategorie:", "Portal:", "Spezial:",
    "Benutzerin:", "Wiktionary:", "File:", "Image:", "Talk:",
    "User:", "Template:", "Help:", "Category:",
)

# ── wiki markup cleaner ──────────────────────────────────────────────────────

try:
    import mwparserfromhell

    def clean_wikitext(wikitext: str) -> str:
        parsed = mwparserfromhell.parse(wikitext)
        text = parsed.strip_code()
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

except ImportError:
    print(
        "Note: mwparserfromhell not installed; using basic regex cleanup.\n"
        "      For cleaner output: pip install mwparserfromhell",
        file=sys.stderr,
    )

    def clean_wikitext(wikitext: str) -> str:  # type: ignore[misc]
        text = wikitext
        # Remove tables
        text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)
        # Remove templates (iterative to handle nesting)
        prev = None
        while prev != text:
            prev = text
            text = re.sub(r"\{\{[^{}]*\}\}", "", text)
        # Remove file/image embeds
        text = re.sub(
            r"\[\[(?:Datei|File|Image|Bild):[^\]]*\]\]", "", text, flags=re.IGNORECASE
        )
        # Keep display text from wikilinks [[target|display]] or [[target]]
        text = re.sub(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]", r"\1", text)
        # External links: keep label, drop bare URLs
        text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
        text = re.sub(r"https?://\S+", "", text)
        # Section headers
        text = re.sub(r"={2,}([^=]+)={2,}", r"\n\1\n", text)
        # Bold / italic
        text = re.sub(r"'{2,}", "", text)
        # HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Table row markup
        text = re.sub(r"^\s*[|!].*$", "", text, flags=re.MULTILINE)
        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# ── helpers ──────────────────────────────────────────────────────────────────

def tag(name: str) -> str:
    return f"{{{MW_NS}}}{name}"


def is_redirect(wikitext: str) -> bool:
    return bool(
        re.match(r"\s*#(?:REDIRECT|redirect|Weiterleitung|weiterleitung)", wikitext)
    )


def is_article_title(title: str) -> bool:
    return not title.startswith(SKIP_PREFIXES)


# ── extraction ───────────────────────────────────────────────────────────────

def load_dialect_map() -> dict[str, str | None]:
    """Load {page_id: dialect_or_None} from als_dialect_map.json."""
    if not DIALECT_MAP_FILE.exists():
        print(
            f"Error: {DIALECT_MAP_FILE.name} not found.\n"
            "Run fetch_dialect_categories.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(DIALECT_MAP_FILE, encoding="utf-8") as f:
        return json.load(f)


def extract_articles() -> None:
    if not DUMP_FILE.exists():
        print(f"Error: dump file not found: {DUMP_FILE}", file=sys.stderr)
        sys.exit(1)

    dialect_map = load_dialect_map()
    print(f"Loaded dialect map: {len(dialect_map):,} tagged page IDs.")

    print(f"Extracting articles from {DUMP_FILE.name} …")
    count = 0
    skipped = 0
    skipped_non_swiss = 0

    with bz2.open(DUMP_FILE, "rb") as f, open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        context = ET.iterparse(f, events=("end",))
        for _, elem in context:
            if elem.tag != tag("page"):
                continue

            # Only main-namespace articles (ns == "0")
            ns_elem = elem.find(tag("ns"))
            if ns_elem is None or ns_elem.text != "0":
                skipped += 1
                elem.clear()
                continue

            title_elem = elem.find(tag("title"))
            id_elem = elem.find(tag("id"))          # first <id> = page id, not revision id
            text_elem = elem.find(f".//{tag('text')}")

            if title_elem is None or id_elem is None:
                skipped += 1
                elem.clear()
                continue

            title = title_elem.text or ""
            page_id = id_elem.text or ""
            wikitext = text_elem.text if text_elem is not None else ""

            if not wikitext:
                skipped += 1
                elem.clear()
                continue

            if is_redirect(wikitext):
                skipped += 1
                elem.clear()
                continue

            if not is_article_title(title):
                skipped += 1
                elem.clear()
                continue

            # Detect dialect via category map and filter non-Swiss articles
            if page_id in dialect_map:
                dialect_value = dialect_map[page_id]
                if dialect_value is None:
                    # Non-Swiss dialect → drop
                    skipped_non_swiss += 1
                    elem.clear()
                    continue
                dialect = dialect_value
            else:
                dialect = "ch"  # no dialect tag → generic Swiss German

            clean_text = clean_wikitext(wikitext)

            if len(clean_text) < 100:
                skipped += 1
                elem.clear()
                continue

            record = {"id": page_id, "title": title, "dialect_before_mapping": dialect, "dialect": DIALECT_TO_CANTON.get(dialect, "ch"), "text": clean_text}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            if count % 1000 == 0:
                print(f"  {count:,} articles written (skipped {skipped:,}) …")

            elem.clear()

    print(f"Done. Extracted {count:,} articles, skipped {skipped:,} (+ {skipped_non_swiss:,} non-Swiss dialects).")
    print(f"Output → {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_articles()
