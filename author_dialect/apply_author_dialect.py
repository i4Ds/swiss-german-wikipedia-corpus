"""Build author->dialect from category tags, then propagate to ALL articles.

Input : <OUT_DIR>/article_authors_from_dump.jsonl   (from parse_creators_from_dump.py)
        <OUT_DIR>/author_declarations.json            (optional, explicit self-tags)
Output: <OUT_DIR>/author_dialect_map.json             (author -> dialect, with purity)
        <OUT_DIR>/articles_dialect_predicted.jsonl     (id,title,dialect,source,confidence)

Author signal preference: top_author_bytes (main content author) > creator.
Both are validated against the human category tags to compute per-author purity.
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_paths import ARTICLE_DUMP, HISTORY_DUMP, EXTRACTED_DIR, RESULTS_DIR

OUT = RESULTS_DIR
ARTS = OUT / "article_authors_from_dump.jsonl"
DECL = OUT / "author_declarations.json"

# Map explicit self-declaration (Hochdeutsch name) -> als category dialect label.
DECL_TO_CAT = {
    "Zürichdeutsch": "Züritüütsch", "Baseldeutsch": "Baseldytsch",
    "Berndeutsch": "Bärndütsch", "Luzernerdeutsch": "Lozäärntüütsch",
    "Aargauerdeutsch": "Aargauerdüütsch", "Senslerdeutsch": "Seislertütsch",
    "St. Gallerdütsch": "St. Gallerdütsch", "Ostschweizerisch": "Ostschwizertütsch",
}


def load_articles():
    arts = []
    with open(ARTS, encoding="utf-8") as f:
        for line in f:
            arts.append(json.loads(line))
    return arts


# Automated / cross-dialect maintenance accounts: never attribute a dialect via
# these. Bots (name contains "bot") plus known maintainers that edit across all
# dialects (caught anyway by the purity filter, listed here for clarity).
MAINTAINER_BLOCKLIST = {"Holder", "Test-tools"}


def is_bot_or_maintainer(user):
    if not user:
        return True
    if user in MAINTAINER_BLOCKLIST:
        return True
    return "bot" in user.lower()


def best_known_author(a, amap):
    """Highest-byte registered author whose dialect is known, skipping bots.

    Falls back through the ranked contributor list, then the creator.
    """
    for user, _bytes in a.get("authors_by_bytes", []):
        if is_bot_or_maintainer(user):
            continue
        if user in amap:
            return user, "author:bytes"
    cr = a.get("creator")
    if cr and not _is_ip(cr) and not is_bot_or_maintainer(cr) and cr in amap:
        return cr, "author:creator"
    return None, None


def build_author_map(arts, purity_min, min_tagged):
    """Majority-vote dialect per author, learned from category-tagged articles.

    An author votes for a dialect for every tagged article where they are the
    top registered (non-bot) content contributor.
    """
    votes = defaultdict(Counter)
    for a in arts:
        d = a.get("category_dialect")
        if not d:
            continue
        author = next((u for u, _ in a.get("authors_by_bytes", [])
                       if not is_bot_or_maintainer(u)), None)
        if author:
            votes[author][d] += 1

    amap = {}
    for author, c in votes.items():
        top, n = c.most_common(1)[0]
        total = sum(c.values())
        purity = n / total
        if total >= min_tagged and purity >= purity_min:
            amap[author] = {"dialect": top, "purity": round(purity, 3),
                            "n_tagged": total, "source": "majority"}

    # Explicit self-declarations fill in / reinforce.
    if DECL.exists():
        for user, info in json.load(open(DECL, encoding="utf-8")).items():
            cat = DECL_TO_CAT.get(info.get("dialect_hg"))
            if cat and user not in amap:
                amap[user] = {"dialect": cat, "purity": 1.0,
                              "n_tagged": 0, "source": "declaration"}
    return amap, votes


def _is_ip(s):
    return all(ch in "0123456789.:abcdefABCDEF" for ch in s) and (
        s.count(".") == 3 or ":" in s)


def predict(arts, amap):
    out = []
    for a in arts:
        cat = a.get("category_dialect")
        if cat:  # already has ground-truth tag
            out.append({**_base(a), "dialect": cat, "source": "category",
                        "confidence": 1.0})
            continue
        author, how = best_known_author(a, amap)
        if author:
            e = amap[author]
            ab = dict(a.get("authors_by_bytes", []))
            total = sum(ab.values()) or 1
            share = round(ab.get(author, 0) / total, 3)
            # Confidence blends how reliable the author is (purity) with how much
            # of THIS article they actually wrote (byte share).
            conf = round(e["purity"] * share, 3)
            out.append({**_base(a), "dialect": e["dialect"], "source": how,
                        "confidence": conf, "author_purity": e["purity"],
                        "author_byte_share": share, "via_author": author,
                        "via_declaration_only": e.get("n_tagged", 0) == 0})
        else:
            out.append({**_base(a), "dialect": None, "source": "unknown",
                        "confidence": 0.0})
    return out


def _base(a):
    return {"id": a["id"], "title": a["title"]}


def main(purity_min, min_tagged):
    arts = load_articles()
    amap, votes = build_author_map(arts, purity_min, min_tagged)
    preds = predict(arts, amap)

    (OUT / "author_dialect_map.json").write_text(
        json.dumps(amap, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(OUT / "articles_dialect_predicted.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # ── report ──────────────────────────────────────────────────────────────
    n = len(preds)
    tagged = sum(1 for p in preds if p["source"] == "category")
    by_author = sum(1 for p in preds if p["source"].startswith("author"))
    unknown = sum(1 for p in preds if p["source"] == "unknown")
    untagged = n - tagged
    print(f"purity_min={purity_min}  min_tagged={min_tagged}")
    print(f"authors in map: {len(amap)} "
          f"({sum(1 for v in amap.values() if v['source']=='declaration')} from declarations)")
    print(f"\narticles total:            {n:,}")
    print(f"  already category-tagged:  {tagged:,} ({tagged/n*100:.1f}%)")
    print(f"  newly tagged by author:   {by_author:,} "
          f"({by_author/untagged*100:.1f}% of the {untagged:,} untagged)")
    print(f"  still unknown:            {unknown:,} ({unknown/n*100:.1f}%)")
    print(f"\ncoverage with a dialect:    {tagged+by_author:,} "
          f"({(tagged+by_author)/n*100:.1f}%)  [was {tagged/n*100:.1f}%]")

    # Honest quality stratification of the author-attributed labels.
    auth = [p for p in preds if p["source"].startswith("author")]
    decl_only = sum(1 for p in auth if p.get("via_declaration_only"))
    hi = sum(1 for p in auth if p["confidence"] >= 0.5
             and not p.get("via_declaration_only"))
    mid = sum(1 for p in auth if 0.25 <= p["confidence"] < 0.5
              and not p.get("via_declaration_only"))
    lo = len(auth) - hi - mid - decl_only
    print("\n--- author-attributed label quality ---")
    print(f"  high  (verified author, conf>=0.5): {hi:,}")
    print(f"  mid   (verified author, 0.25-0.5):  {mid:,}")
    print(f"  low   (verified author, conf<0.25): {lo:,}")
    print(f"  declaration-only (UNVERIFIED):      {decl_only:,}")
    print(f"  => category + high-confidence only: {tagged+hi:,} "
          f"({(tagged+hi)/n*100:.1f}%)")

    dist = Counter(p["dialect"] for p in preds if p["dialect"])
    print("\n--- predicted dialect distribution (all sources) ---")
    for d, c in dist.most_common():
        print(f"  {c:6,}  {d}")
    print(f"\n-> {OUT/'author_dialect_map.json'}")
    print(f"-> {OUT/'articles_dialect_predicted.jsonl'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--purity-min", type=float, default=0.8)
    ap.add_argument("--min-tagged", type=int, default=2)
    a = ap.parse_args()
    main(a.purity_min, a.min_tagged)
