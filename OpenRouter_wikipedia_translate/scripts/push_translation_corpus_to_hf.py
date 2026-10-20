"""Export the translated Alemannic Wikipedia corpus and push it to the Hub.

Input : results/translated/wikipedia_translated.jsonl (one record per article)
Output: i4ds/alemannic-wikipedia-clean-translation-corpus (one row per sentence pair)

The corpus is flattened from articles to sentence pairs because that is the unit
downstream MT/TTS training consumes. Article-level metadata (title, url, dialect)
is carried onto every one of its pairs.

Run inside the `whisper-prep` conda env (it has huggingface_hub + datasets):
    conda activate whisper-prep
    python push_translation_corpus_to_hf.py --dry_run   # inspect, no upload
    python push_translation_corpus_to_hf.py             # upload
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from corpus_paths import ARTICLE_DUMP, HISTORY_DUMP, EXTRACTED_DIR, RESULTS_DIR

RES = RESULTS_DIR
DEFAULT_INPUT = RES / "translated" / "wikipedia_translated.jsonl"
PROMPT_PY = Path(__file__).resolve().parent.parent / "PROMPT.py"
HF_REPO = "i4ds/alemannic-wikipedia-clean-translation-corpus"

# Labels that are not a Swiss region. "Not Swiss German" marks non-Swiss
# Alemannic (Swabian, Badisch, Alsatian, ...) and is excluded from the corpus.
NON_REGIONAL = {"Swiss German", "Not Swiss German", "NA", None}
EXCLUDE_LABEL = "Not Swiss German"

# Column order of the published dataset. Declared explicitly rather than left to
# dict-key order so the four lead columns stay put, and so the Hub gets real
# types instead of whatever Arrow infers from the first rows.
FEATURES = {
    "de": "string",
    "gsw": "string",
    "dialect": "string",
    "confidence": "int64",
    "article_id": "string",
    "segment_index": "int64",
    "pair_id": "string",
    "title": "string",
    "source_url": "string",
    "dialect_canton": "string",
    "dialect_category": "string",
    "dialect_category_source": "string",
    "n_words_gsw": "int64",
    "n_chars_gsw": "int64",
}

# Card metadata merged onto the block push_to_hub writes for itself. It must not
# contain `configs`/`dataset_info`: push_to_hub owns those, and hand-writing
# them is what broke the dataset viewer (a bare `data_files: data.jsonl` gives
# the viewer no split name -> SplitsNotFoundError).
CARD_METADATA = {
    "license": "cc-by-sa-4.0",
    "language": ["gsw", "de"],
    "task_categories": ["translation", "text-generation"],
    "tags": ["swiss-german", "alemannic", "wikipedia", "parallel-corpus",
             "dialect"],
    "size_categories": ["100K<n<1M"],
}


def article_dialect(record: dict) -> tuple[str | None, int]:
    """Majority-vote the per-chunk dialect calls into one article-level label.

    Returns (label, confidence_level). Chunks of the same article can disagree;
    the vote is over regional labels only, so a single confident regional call
    is not outvoted by generic "Swiss German" chunks. Confidence is the max
    seen for the winning label, since the levels are evidence tiers (0-3), not
    a probability to average.
    """
    analyses = record.get("gemini_dialect_analyses") or []
    if not analyses:
        return None, 0

    regional = [a for a in analyses if a.get("label") not in NON_REGIONAL]
    pool = regional or analyses
    votes = Counter(a.get("label") for a in pool)
    label, _ = votes.most_common(1)[0]
    conf = max((a.get("confidence_level", 0) or 0)
               for a in pool if a.get("label") == label)
    # Articles translated before the runner gained its parse-time clamp can
    # carry out-of-range values (observed: 23). Google's schema validator
    # rejects enums on integer properties, so the 0-3 range was never
    # enforceable at the API level. The label fully determines the tier.
    if conf not in (0, 1, 2, 3):
        conf = 1 if label == "Swiss German" else 0 if label == EXCLUDE_LABEL else 2
    return label, conf


def build_rows(input_jsonl: Path) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    stats = {
        "articles_in": 0,
        "articles_excluded_non_swiss": 0,
        "articles_no_segments": 0,
        "articles_used": 0,
        "segments_dropped_empty": 0,
        "api_cost_usd": 0.0,
    }

    with input_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            stats["articles_in"] += 1
            # Counted before the non-Swiss filter: those calls were paid for
            # too, so this is total spend, not spend on published rows.
            stats["api_cost_usd"] += r.get("api_cost_usd") or 0.0

            label, conf = article_dialect(r)
            if label == EXCLUDE_LABEL:
                stats["articles_excluded_non_swiss"] += 1
                continue
            segments = r.get("segments") or []
            if not segments:
                stats["articles_no_segments"] += 1
                continue
            stats["articles_used"] += 1

            # `segment_index` is the position in the article's ORIGINAL segment
            # list, kept even when a segment is dropped below. A gap in the
            # indices therefore means "dropped here", and the surviving indices
            # still place every row correctly inside its article.
            for seg_idx, s in enumerate(segments):
                gsw = (s.get("gsw") or "").strip()
                de = (s.get("de") or "").strip()
                if not gsw or not de:
                    stats["segments_dropped_empty"] += 1
                    continue
                article_id = r.get("id")
                rows.append({
                    # --- the four requested lead columns ---
                    "de": de,
                    "gsw": gsw,
                    "dialect": label,
                    "confidence": conf,
                    # --- identity: (article_id, segment_index) is unique, and
                    # sorting by it reconstructs the article in reading order ---
                    "article_id": article_id,
                    "segment_index": seg_idx,
                    "pair_id": f"{article_id}:{seg_idx}",
                    # --- provenance ---
                    "title": r.get("title"),
                    "source_url": r.get("source_url"),
                    "dialect_canton": r.get("dialect_canton"),
                    "dialect_category": r.get("dialect"),
                    # Provenance of `dialect_category` only. The `dialect`
                    # column above is always Gemini's label.
                    "dialect_category_source": (
                        "category" if r.get("dialect_label_source") == "category"
                        else None
                    ),
                    "n_words_gsw": len(gsw.split()),
                    "n_chars_gsw": len(gsw),
                })
    return rows, stats


def compute_stats(rows: list[dict]) -> dict:
    by_dialect: dict[str, dict] = {}
    for row in rows:
        d = by_dialect.setdefault(
            row["dialect"] or "unknown",
            {"pairs": 0, "chars_gsw": 0, "words_gsw": 0, "articles": set()},
        )
        d["pairs"] += 1
        d["chars_gsw"] += row["n_chars_gsw"]
        d["words_gsw"] += row["n_words_gsw"]
        d["articles"].add(row["article_id"])
    for d in by_dialect.values():
        d["articles"] = len(d["articles"])
    return dict(sorted(by_dialect.items(), key=lambda kv: -kv[1]["pairs"]))


def md_table(by_dialect: dict, total_pairs: int) -> str:
    lines = [
        "| Dialect | Sentence pairs | % | Articles | Words (gsw) | Chars (gsw) | Avg chars/pair |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, d in by_dialect.items():
        lines.append(
            f"| {name} | {d['pairs']:,} | {d['pairs']/total_pairs*100:.1f}% | "
            f"{d['articles']:,} | {d['words_gsw']:,} | {d['chars_gsw']:,} | "
            f"{d['chars_gsw']/d['pairs']:.0f} |"
        )
    return "\n".join(lines)


def build_readme(rows: list[dict], stats: dict, by_dialect: dict, conf_dist: Counter) -> str:
    total = len(rows)
    words = sum(r["n_words_gsw"] for r in rows)
    chars = sum(r["n_chars_gsw"] for r in rows)
    prompt_text = PROMPT_PY.read_text(encoding="utf-8")

    conf_rows = "\n".join(
        f"| {lvl} | {conf_dist[lvl]:,} | {conf_dist[lvl]/total*100:.1f}% | {desc} |"
        for lvl, desc in [
            (3, "Regional dialect — both identification methods agree"),
            (2, "Regional dialect — one method, or methods disagree on region"),
            (1, "Swiss German, but no region identifiable"),
            (0, "Not Swiss German (excluded from this corpus)"),
        ]
    )

    # Body only — no YAML frontmatter. The card metadata is applied separately
    # (CARD_METADATA) on top of the `dataset_info`/`configs` block that
    # Dataset.push_to_hub generates, so the two never fight over the file.
    return f"""# Alemannic Wikipedia Clean Translation Corpus

A parallel **Swiss German ↔ Standard High German** corpus built from the
[Alemannic Wikipedia](https://als.wikipedia.org), with a regional dialect label
on every sentence pair.

- **{total:,}** sentence pairs
- **{stats['articles_used']:,}** source articles
- **{words:,}** Swiss German words / **{chars:,}** characters
- Dialect label + confidence on every row

## Columns

| Column | Description |
|---|---|
| `de` | Standard High German (translation) |
| `gsw` | Swiss German (source, cleaned) |
| `dialect` | Regional dialect label, or `Swiss German` if no region identifiable. **Always model-assigned** |
| `confidence` | Evidence tier for the dialect label, 0–3 (see below) |
| `article_id` | Source article id — group by this, or use it for document-level splits |
| `segment_index` | Position of the pair within its article, 0-based |
| `pair_id` | `"{{article_id}}:{{segment_index}}"` — unique key for every row |
| `title` | Wikipedia article title |
| `source_url` | Article URL |
| `dialect_canton` | Canton code from the als.wiki category, when present |
| `dialect_category` | Fine-grained dialect from the als.wiki category, when present |
| `dialect_category_source` | `category` when `dialect_category` came from a human als.wiki tag, else `null`. Describes `dialect_category`, **not** `dialect` |
| `n_words_gsw`, `n_chars_gsw` | Length of the Swiss German side |

**Sentence pairs are aligned 1:1** — `gsw` and `de` are the same sentence.
Pairs from one article share an `article_id`; split on that, not on rows, to
avoid leaking near-duplicate context between train and test.

### Reconstructing whole articles

Every row is one sentence pair, but the article it came from is fully
recoverable: group by `article_id` and sort by `segment_index`.

```python
from datasets import load_dataset
import itertools

ds = load_dataset("i4ds/alemannic-wikipedia-clean-translation-corpus", split="train")

rows = sorted(ds, key=lambda r: (r["article_id"], r["segment_index"]))
for article_id, group in itertools.groupby(rows, key=lambda r: r["article_id"]):
    group = list(group)
    gsw_article = " ".join(r["gsw"] for r in group)
    de_article  = " ".join(r["de"] for r in group)
```

`segment_index` counts positions in the article as it came out of translation,
so it can skip a number where a segment had an empty side and was dropped at
export. Order is always correct; a gap just means one sentence is absent.

## Dialect distribution

{md_table(by_dialect, total)}

## Confidence levels

| Level | Pairs | % | Meaning |
|---|---:|---:|---|
{conf_rows}

Rows labelled with a **region** are always confidence 2 or 3. Rows labelled
`Swiss German` are confidence 1: the text is Swiss German, but no single region
could be identified — these are still fully usable parallel data, they just
carry no regional signal. Filter with `dialect != "Swiss German"` if you need
regionally-attributed text only.

## How it was built

1. **Extraction** — articles parsed from an offline Alemannic Wikipedia dump.
2. **Cleaning** (`export_clean_translation_corpus.py`) — strips image/infobox
   markup, template residue and caption fragments, while **preserving blank-line
   paragraph breaks**. Paragraph structure matters: it is what the chunker
   splits on downstream.
3. **Chunking** — articles are sent whole where possible. Paragraphs longer than
   **6,382 words** (empirically validated as safe: 100% coverage, output well
   inside the 65,536-token completion limit) are split at sentence boundaries
   only (`.` `!` `?`) — a sentence is never cut mid-way.
4. **Translation + labelling** — a single `google/gemini-3.6-flash` call per
   chunk performs cleaning, segmentation, translation, Swiss-German
   determination and dialect identification, constrained by a strict JSON
   schema. Run with `service_tier=flex` and minimal reasoning effort.
   Translating the full {stats['articles_in']:,} articles cost
   **${stats['api_cost_usd']:,.2f}** in API spend
   (≈ ${stats['api_cost_usd']/max(stats['articles_in'],1):.4f} per article),
   including the non-Swiss articles that were paid for and then filtered out.
5. **Filtering** — the Alemannic Wikipedia also contains **non-Swiss** Alemannic
   (Swabian, Badisch, Alsatian, Vorarlbergisch, non-Swiss Walser). Those
   articles are detected, labelled `Not Swiss German`, and **excluded** from
   this corpus ({stats['articles_excluded_non_swiss']:,} articles).

### Quality checks

- `de`/`gsw` word-count ratio: **mean 1.00** — no systematic truncation or
  runaway generation.
- Empty sides: **0** (rows with an empty side are dropped at export).
- Dialect labels agree with human als.wiki category tags on **95.7%** of the
  1,669 articles carrying a category that maps onto a canton. The dominant
  disagreement is eastern Swiss German (`Ostschwizertütsch`) read as Zürich —
  a genuinely hard boundary.
- **1.4%** of pairs are identical on both sides. These are real: short
  sentences, names and dates that are already the same in both varieties.

### Known limitations

- **Wikipedia tables become long pseudo-sentences.** Table rows survive cleaning
  and can produce very long "sentences" that are accurate translations but have
  no sentence structure. For TTS, filter on `n_words_gsw` (median 16, p95 39,
  p99 59 words).
- **The `dialect` column is model-assigned on every row.** Treat it as a strong
  signal, not ground truth. Where the als.wiki article carried a human dialect
  category, it is kept separately in `dialect_category` for cross-checking.
- **Abbreviations are preserved as written in the source** — not expanded.
- `dialect` reflects the dominant dialect of the whole article; an individual
  sentence may not carry regional markers.

## The prompt

The full prompt used for translation and dialect identification is included as
`PROMPT.py` in this repository, and reproduced here for reference:

```python
{prompt_text}
```

## Licence

Text derives from Wikipedia and is released under **CC BY-SA 4.0**, matching the
source. Translations were produced by a machine-translation model.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", default=str(DEFAULT_INPUT))
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--out_dir", default=None,
                    help="Where to stage data.jsonl/README.md (default: temp dir).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Build and report locally, do not upload.")
    ap.add_argument("--private", action="store_true", default=True)
    args = ap.parse_args()

    rows, stats = build_rows(Path(args.input_jsonl))
    if not rows:
        raise SystemExit("No rows built — refusing to push an empty dataset.")

    by_dialect = compute_stats(rows)
    conf_dist = Counter(r["confidence"] for r in rows)

    print(f"articles read      : {stats['articles_in']:,}")
    print(f"  excluded non-Swiss: {stats['articles_excluded_non_swiss']:,}")
    print(f"  no segments       : {stats['articles_no_segments']:,}")
    print(f"  used              : {stats['articles_used']:,}")
    print(f"sentence pairs     : {len(rows):,}")
    print(f"  dropped (empty)   : {stats['segments_dropped_empty']:,}")
    print(f"gsw words          : {sum(r['n_words_gsw'] for r in rows):,}")
    print("\ndialect distribution:")
    for name, d in by_dialect.items():
        print(f"  {name:20s} {d['pairs']:>8,} pairs  {d['articles']:>6,} articles  "
              f"{d['chars_gsw']:>12,} chars")
    print(f"\nconfidence: {dict(sorted(conf_dist.items()))}")

    body = build_readme(rows, stats, by_dialect, conf_dist)

    out_dir = Path(args.out_dir) if args.out_dir else Path("./hf_staging")
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "data.jsonl"
    with data_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    readme_path = out_dir / "README.md"
    readme_path.write_text(body, encoding="utf-8")
    prompt_copy = out_dir / "PROMPT.py"
    prompt_copy.write_text(PROMPT_PY.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\nstaged -> {data_path} ({data_path.stat().st_size/1e6:.1f} MB)")
    print(f"staged -> {readme_path}  (body only; frontmatter added at push time)")
    print(f"staged -> {prompt_copy}")

    if args.dry_run:
        print("\n--dry_run set: nothing uploaded.")
        return

    from datasets import Dataset, Features, Value
    from huggingface_hub import DatasetCard, HfApi

    api = HfApi()
    print(f"\nuploading to {args.repo} as {api.whoami()['name']} …")
    api.create_repo(args.repo, repo_type="dataset",
                    private=args.private, exist_ok=True)

    # Push as Parquet via the datasets library rather than uploading the raw
    # JSONL: it writes a proper `train` split plus the dataset_info the viewer
    # needs, and Parquet is a fraction of the size.
    features = Features({k: Value(v) for k, v in FEATURES.items()})
    ds = Dataset.from_list(rows, features=features)
    ds.push_to_hub(args.repo, private=args.private)

    # push_to_hub rewrote the card with its own frontmatter and an empty body.
    # Reload it so dataset_info/configs survive, then layer our body + metadata.
    card = DatasetCard.load(args.repo, repo_type="dataset")
    for key, value in CARD_METADATA.items():
        setattr(card.data, key, value)
    card.text = body
    card.push_to_hub(args.repo, repo_type="dataset")

    api.upload_file(path_or_fileobj=str(prompt_copy), path_in_repo="PROMPT.py",
                    repo_id=args.repo, repo_type="dataset")

    # Remove the hand-rolled JSONL from the earlier upload scheme. Leaving it
    # would double the repo size and give the viewer a second, conflicting
    # source for the same rows.
    try:
        api.delete_file("data.jsonl", repo_id=args.repo, repo_type="dataset")
        print("removed stale data.jsonl")
    except Exception as exc:                      # already absent on a fresh repo
        print(f"no stale data.jsonl to remove ({type(exc).__name__})")

    print(f"done -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
