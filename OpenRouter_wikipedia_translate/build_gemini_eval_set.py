"""Build a ~300-article eval set for the Gemini dialect-detection experiment.

Goal: compare the LLM's dialect identification against our two label sources:
  - category      (human dialect category on als.wikipedia -> ground truth)
  - author:bytes  (inferred from the main content author -> to be validated)

Sample design (seed fixed, stratified by dialect within each group):
  150 category rows
  150 author rows: 75 high (conf>=0.5), 30 mid (0.25-0.5),
                   25 low (<0.25), 20 declaration-only

Input : clean_translation_corpus.jsonl  (already cleaned, runner-compatible)
Output: <OUT_DIR>/gemini_eval/eval_set_300.jsonl  (same schema + "eval_tier")

Quotas are upper bounds. The current category-only cleaned corpus has no author
labels, so only the category stratum is populated (up to 150 articles). Author
strata require input carrying the historical author-label metadata.
"""
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_paths import ARTICLE_DUMP, HISTORY_DUMP, EXTRACTED_DIR, RESULTS_DIR

OUT_DIR = RESULTS_DIR
CORPUS = OUT_DIR / "clean_translation_corpus.jsonl"
EVAL_DIR = OUT_DIR / "gemini_eval"
EVAL_FILE = EVAL_DIR / "eval_set_300.jsonl"

SEED = 42
MIN_WORDS, MAX_WORDS = 50, 1200  # skip stubs and monster articles
PER_DIALECT_CAP = 40

QUOTA = {
    "category": 150,
    "author_high": 75,
    "author_mid": 30,
    "author_low": 25,
    "author_decl_only": 20,
}


def tier_of(row):
    src = row.get("dialect_label_source") or ""
    if src == "category":
        return "category"
    if src.startswith("author"):
        if row.get("dialect_declaration_only"):
            return "author_decl_only"
        conf = row.get("dialect_confidence", 0.0)
        if conf >= 0.5:
            return "author_high"
        if conf >= 0.25:
            return "author_mid"
        return "author_low"
    return None


def stratified_sample(rows, k, rng):
    """Sample k rows, capping each dialect at PER_DIALECT_CAP where possible."""
    by_dialect = defaultdict(list)
    for r in rows:
        by_dialect[r["dialect"]].append(r)
    picked = []
    for d, items in by_dialect.items():
        rng.shuffle(items)
        picked.extend(items[:PER_DIALECT_CAP])
    if len(picked) > k:
        picked = rng.sample(picked, k)
    elif len(picked) < k:
        pool = [r for items in by_dialect.values() for r in items if r not in picked]
        rng.shuffle(pool)
        picked.extend(pool[: k - len(picked)])
    return picked


def main():
    rng = random.Random(SEED)
    buckets = defaultdict(list)
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not (MIN_WORDS <= row.get("word_count", 0) <= MAX_WORDS):
                continue
            t = tier_of(row)
            if t:
                buckets[t].append(row)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for tier, k in QUOTA.items():
        pool = buckets.get(tier, [])
        take = stratified_sample(pool, min(k, len(pool)), rng)
        for r in take:
            r["eval_tier"] = tier
        out_rows.extend(take)
        print(f"{tier:18s} pool={len(pool):6,}  sampled={len(take)}")

    rng.shuffle(out_rows)
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(out_rows)} rows -> {EVAL_FILE}")
    print("dialect distribution:")
    for d, c in Counter(r["dialect"] for r in out_rows).most_common():
        print(f"  {c:4d}  {d}")


if __name__ == "__main__":
    main()
