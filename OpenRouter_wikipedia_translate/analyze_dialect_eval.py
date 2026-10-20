"""Compare Gemini's dialect identification against our category / author labels.

Input : the wikipedia_translated.jsonl produced by run_openrouter_wikipedia_translate.py
        on the eval_set_300.jsonl sample.

For each article, regional labels from the per-chunk Gemini dialect analyses are
majority-voted and compared with our label mapped to the same 7 macro classes.
Generic and non-Swiss labels are excluded from scoring. Confidence is averaged
across chunks on the prompt's 0-3 evidence scale.

Key readout:
  - accuracy on eval_tier == "category"  -> how good is the LLM (human tags = truth)
  - accuracy on author tiers             -> how good are the AUTHOR labels,
                                            assuming the LLM is as good as measured
                                            on the category rows.
"""
import argparse
import json
from collections import Counter, defaultdict

# our fine dialect label -> Gemini's 7 macro classes
MACRO = {
    "Baseldytsch": "Basel", "Baselbieterdütsch": "Basel", "Soledurnerdütsch": "Basel",
    "Züritüütsch": "Zürich", "Aargauerdüütsch": "Zürich", "Freiämtertütsch": "Zürich",
    "Bärndütsch": "Bern", "Seislertütsch": "Bern",
    "Ostschwizertütsch": "St. Gallen", "St. Gallerdütsch": "St. Gallen",
    "Glarnertüütsch": "St. Gallen",
    "Wallisertiitsch": "Wallis", "Walsertiitsch": "Wallis",
    "Lozäärntüütsch": "Luzern", "Innerschwyzerisch": "Luzern", "Schwyzerdütsch": "Luzern",
    "Pündnertütsch": "Graubünden",
    # "Schwizerdütsch" is generic -> excluded from scoring
}


# Labels that are not a regional dialect and therefore cannot be scored against
# our 7-class macro labels.
NON_REGIONAL = {"NA", "Swiss German", "Not Swiss German"}


def gemini_macro(analyses):
    """Majority-vote the per-chunk {label, confidence_level} calls (V3 prompt schema).

    Returns (top_label, vote_share, mean_confidence_level). confidence_level is on
    the prompt's 0-3 scale (0=not Swiss German, 1=Swiss German but no region,
    2=one method, 3=both methods agree), not the old 0-100 certainty scale.
    """
    if not analyses:
        return None, 0.0, 0.0
    votes = Counter()
    confs = []
    for a in analyses:
        confs.append(a.get("confidence_level", 0))
        label = a.get("label")
        if label and label not in NON_REGIONAL:
            votes[label] += 1
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    if not votes:
        return None, 0.0, mean_conf
    top, n = votes.most_common(1)[0]
    return top, n / len(analyses), mean_conf


def main(path, min_certainty):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    per_tier = defaultdict(lambda: {"n": 0, "scored": 0, "hit": 0, "conf": []})
    confusion = Counter()
    misses = []

    for r in rows:
        tier = r.get("eval_tier") or ("category" if r.get("dialect_label_source") == "category"
                                      else "author")
        truth = MACRO.get(r.get("dialect"))
        pred, prob, cert = gemini_macro(r.get("gemini_dialect_analyses", []))
        s = per_tier[tier]
        s["n"] += 1
        if truth is None or pred is None or cert < min_certainty:
            continue
        s["scored"] += 1
        s["conf"].append(cert)
        if pred == truth:
            s["hit"] += 1
        else:
            misses.append((tier, r["title"], r.get("dialect"), truth, pred,
                           round(prob, 2), round(cert), r.get("author"),
                           r.get("dialect_confidence")))
        confusion[(truth, pred)] += 1

    print(f"articles: {len(rows)}   (scoring threshold: confidence_level >= {min_certainty}, scale 0-3)\n")
    print(f"{'tier':18s} {'n':>4} {'scored':>6} {'agree':>6} {'agree%':>7} {'mean_conf':>9}")
    order = ["category", "author_high", "author_mid", "author_low", "author_decl_only"]
    for tier in order + sorted(set(per_tier) - set(order)):
        if tier not in per_tier:
            continue
        s = per_tier[tier]
        pct = s["hit"] / s["scored"] * 100 if s["scored"] else 0.0
        mc = sum(s["conf"]) / len(s["conf"]) if s["conf"] else 0.0
        print(f"{tier:18s} {s['n']:>4} {s['scored']:>6} {s['hit']:>6} {pct:>6.1f}% {mc:>9.2f}")

    print("\n--- confusion (truth -> gemini), scored rows ---")
    labels = sorted({t for t, _ in confusion} | {p for _, p in confusion})
    width = max((len(l) for l in labels), default=0) + 1
    print(" " * width + "".join(f"{l[:9]:>10}" for l in labels))
    for t in labels:
        print(f"{t:<{width}}" + "".join(f"{confusion.get((t, p), 0):>10}" for p in labels))

    print(f"\n--- disagreements ({len(misses)}) ---")
    for tier, title, fine, truth, pred, prob, cert, author, lconf in sorted(misses):
        print(f"  [{tier}] {title!r}: ours={truth} ({fine}, label_conf={lconf}) "
              f"gemini={pred} (p={prob}, cert={cert}) author={author}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("translated_jsonl")
    ap.add_argument("--min_certainty", type=float, default=0,
                    help="Skip rows where Gemini's mean confidence_level (0-3 scale) is below this.")
    a = ap.parse_args()
    main(a.translated_jsonl, a.min_certainty)
