"""Flatten the article-level translation dump into one segment per line.

Input : wikipedia_translated.jsonl   (one article per line, segments nested)
Output: wikipedia_translated_segments.jsonl (one sentence pair per line)

The runner writes one long line per article because that is the unit it
checkpoints and resumes on — that file stays as-is, it is the raw run output.
This produces the per-segment view for reading, grepping and diffing.

Every row carries `article_id` + `segment_index`, so the original article is
reconstructed exactly by grouping on `article_id` and sorting on
`segment_index`. `--verify` checks that round trip against the source before
declaring success.

    python flatten_segments_jsonl.py            # write + verify
    python flatten_segments_jsonl.py --verify   # verify an existing output
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from corpus_paths import ARTICLE_DUMP, HISTORY_DUMP, EXTRACTED_DIR, RESULTS_DIR

RES = RESULTS_DIR
DEFAULT_INPUT = RES / "translated" / "wikipedia_translated.jsonl"
DEFAULT_OUTPUT = RES / "translated" / "wikipedia_translated_segments.jsonl"

# Article-level fields copied onto each of its segments. `api_cost_usd` is
# deliberately absent: it is a property of the API call, not of the text, and
# repeating it on every segment invites it being mistaken for a per-row cost.
CARRY = ("title", "source_url", "dialect", "dialect_canton",
         "dialect_label_source", "dialect_confidence", "author")


def flatten(record: dict) -> list[dict]:
    """One row per segment, in article order."""
    article_id = record.get("id")
    analyses = record.get("gemini_dialect_analyses") or []
    rows = []
    for idx, seg in enumerate(record.get("segments") or []):
        row = {
            "pair_id": f"{article_id}:{idx}",
            "article_id": article_id,
            "segment_index": idx,
            "gsw": seg.get("gsw"),
            "de": seg.get("de"),
        }
        row.update({k: record.get(k) for k in CARRY})
        row["gemini_dialect_analyses"] = analyses
        rows.append(row)
    return rows


def verify(src: Path, out: Path) -> None:
    """Reconstruct articles from the flat file and compare to the source."""
    rebuilt: dict[str, dict[int, dict]] = defaultdict(dict)
    with out.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rebuilt[r["article_id"]][r["segment_index"]] = {
                "gsw": r["gsw"], "de": r["de"]}

    checked = mismatched = empty_articles = 0
    with src.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            original = rec.get("segments") or []
            if not original:
                empty_articles += 1
                continue
            got = rebuilt.get(rec.get("id"), {})
            restored = [got[i] for i in sorted(got)]
            if restored != [{"gsw": s.get("gsw"), "de": s.get("de")}
                            for s in original]:
                mismatched += 1
                if mismatched <= 3:
                    print(f"  MISMATCH in article {rec.get('id')}")
            checked += 1

    print(f"verify: {checked:,} articles reconstructed, "
          f"{mismatched:,} mismatched, {empty_articles:,} had no segments")
    if mismatched:
        raise SystemExit("round trip FAILED — output does not rebuild the source")
    print("verify: round trip is exact ✓")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", default=str(DEFAULT_INPUT))
    ap.add_argument("--output_jsonl", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--verify", action="store_true",
                    help="Only verify an existing output file.")
    args = ap.parse_args()

    src, out = Path(args.input_jsonl), Path(args.output_jsonl)

    if not args.verify:
        articles = segments = 0
        with src.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fo:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                articles += 1
                for row in flatten(json.loads(line)):
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                    segments += 1
        print(f"read  {articles:,} articles")
        print(f"wrote {segments:,} segment rows -> {out} "
              f"({out.stat().st_size/1e6:.1f} MB)")

    verify(src, out)


if __name__ == "__main__":
    main()
