"""Join article TEXT with the inferred author + dialect into one clean JSONL.

Left  : als_articles.jsonl              (text + category dialect, from extract_als_articles.py)
Right : articles_dialect_predicted.jsonl (dialect/author/confidence, this module)
Key   : page id

The text file's own category tag (dialect_before_mapping != "ch") wins; the
author-inferred prediction only fills in articles without a category tag.

Output: articles_with_author_dialect.jsonl — one object per article:
    {
      "id", "title", "text",
      "dialect",        # inferred (or category) dialect name, may be null
      "dialect_canton", # 2-letter canton code ("ch" = unknown/unmappable)
      "author",         # the author the dialect was inferred from, may be null
      "confidence",     # 0..1  (1.0 for human category tags)
      "label_source",   # "category" | "author:bytes" | "author:creator" | "unknown"
      "declaration_only" # True if dialect came only from an unverified self-declaration
    }
Only articles that have text are emitted (the text file is the corpus of record).
"""
import json
import os
import sys
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_paths import ARTICLE_DUMP, HISTORY_DUMP, EXTRACTED_DIR, RESULTS_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract_als_articles import DIALECT_TO_CANTON, OUTPUT_FILE as ARTICLES_FILE

OUT_DIR = RESULTS_DIR
TEXT_FILE = ARTICLES_FILE
PRED_FILE = OUT_DIR / "articles_dialect_predicted.jsonl"
OUT_FILE = OUT_DIR / "articles_with_author_dialect.jsonl"


def load_predictions():
    preds = {}
    with open(PRED_FILE, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            preds[p["id"]] = p
    return preds


def main():
    preds = load_predictions()
    print(f"Loaded {len(preds):,} predictions.")

    n = matched = by_category = by_author = 0
    with open(TEXT_FILE, encoding="utf-8") as fin, \
         open(OUT_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            a = json.loads(line)
            n += 1
            cat = a.get("dialect_before_mapping")
            p = preds.get(a["id"])

            if cat and cat != "ch":
                # The text file's own category tag is ground truth.
                dialect, author, label_source = cat, None, "category"
                confidence, decl_only = 1.0, False
                by_category += 1
            elif p is not None and p.get("dialect") and p["source"].startswith("author"):
                dialect = p["dialect"]
                author = p.get("via_author")
                confidence = p.get("confidence", 0.0)
                label_source = p["source"]
                decl_only = p.get("via_declaration_only", False)
                by_author += 1
            else:
                dialect = author = label_source = None
                confidence, decl_only = 0.0, False

            if dialect:
                matched += 1
            rec = {
                "id": a["id"],
                "title": a["title"],
                "text": a["text"],
                "dialect": dialect,
                "dialect_canton": DIALECT_TO_CANTON.get(dialect, "ch"),
                "author": author,
                "confidence": confidence,
                "label_source": label_source,
                "declaration_only": decl_only,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {n:,} articles -> {OUT_FILE}")
    print(f"  with a dialect label: {matched:,} ({matched/n*100:.1f}%)")
    print(f"    from category tag:  {by_category:,}")
    print(f"    from author signal: {by_author:,}")


if __name__ == "__main__":
    main()
