"""Portable storage defaults shared by the Wikipedia workflow entry points."""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get(
    "WIKI_DATA_ROOT", os.environ.get("DATA_ROOT", str(Path.home() / ".local/share/swiss-german-wikipedia-corpus"))
)).expanduser().resolve()
RAW_DIR = DATA_ROOT / "raw"
EXTRACTED_DIR = DATA_ROOT / "intermediate/extracted"
RESULTS_DIR = Path(os.environ.get("OUT_DIR", str(DATA_ROOT / "results"))).expanduser().resolve()
ARTICLE_DUMP = Path(os.environ.get("WIKI_ARTICLE_DUMP", str(RAW_DIR / "alswiki-latest-pages-articles.xml.bz2"))).expanduser()
HISTORY_DUMP = Path(os.environ.get("WIKI_HISTORY_DUMP", str(RAW_DIR / "alswiki-latest-stub-meta-history.xml.gz"))).expanduser()
