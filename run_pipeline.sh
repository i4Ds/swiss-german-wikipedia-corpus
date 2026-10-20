#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON:-python3}"
"$PYTHON_BIN" -c 'from corpus_paths import ARTICLE_DUMP; assert ARTICLE_DUMP.is_file(), f"Set WIKI_ARTICLE_DUMP or place the dump at {ARTICLE_DUMP}"'
MAP_FILE="$("$PYTHON_BIN" -c 'from corpus_paths import EXTRACTED_DIR; print(EXTRACTED_DIR / "als_dialect_map.json")')"
if [[ ! -f "$MAP_FILE" || "${REFRESH_CATEGORIES:-0}" == "1" ]]; then
    "$PYTHON_BIN" fetch_dialect_categories.py
fi
"$PYTHON_BIN" extract_als_articles.py
if [[ "${RUN_AUTHOR_EXPERIMENTS:-0}" == "1" ]]; then
    "$PYTHON_BIN" author_dialect/parse_creators_from_dump.py
    if [[ "${REFRESH_AUTHOR_DECLARATIONS:-0}" == "1" ]]; then
        "$PYTHON_BIN" author_dialect/scrape_author_dialect.py declarations
    fi
    "$PYTHON_BIN" author_dialect/apply_author_dialect.py
    "$PYTHON_BIN" author_dialect/join_text_with_dialect.py
fi
"$PYTHON_BIN" author_dialect/export_clean_translation_corpus.py
