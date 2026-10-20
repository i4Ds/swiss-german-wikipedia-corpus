"""Export a cleaned ALS Wikipedia corpus for later High German translation.

Input:
    als_articles.jsonl

The author-based dialect join is intentionally NOT used: inferring an article's
dialect from its main contributing author agreed with human category tags only
25-33% of the time, so those labels are dropped. The only dialect label kept
here is the human-applied Wikipedia category tag; Gemini classifies the rest at
translation time.

Output:
    clean_translation_corpus.jsonl
    clean_translation_corpus.manifest.json

The output keeps one JSON object per source article. It does not call Gemini or
any other translation service.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_paths import ARTICLE_DUMP, HISTORY_DUMP, EXTRACTED_DIR, RESULTS_DIR
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract_als_articles import OUTPUT_FILE as ARTICLES_FILE

OUT_DIR = RESULTS_DIR
INPUT_FILE = ARTICLES_FILE
OUTPUT_FILE = OUT_DIR / "clean_translation_corpus.jsonl"
MANIFEST_FILE = OUT_DIR / "clean_translation_corpus.manifest.json"
HF_EXPORT_DIR = OUT_DIR / "hf_clean_translation_corpus"
HF_DATA_FILE = HF_EXPORT_DIR / "data.jsonl"
HF_README_FILE = HF_EXPORT_DIR / "README.md"

MIN_CHARS = 100
MEDIA_MARK = " __MEDIA_MARK__ "

MAGIC_WORD_RE = re.compile(r"__[^_\s]+__", re.IGNORECASE)
NON_LATIN_SCRIPT_RE = re.compile(
    r"[\u0370-\u03FF\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF"
    r"\u0900-\u097F\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]+"
)
MEDIA_PREFIX_RE = re.compile(
    r"""(?ix)
    (?:
        (?:links|rechts|left|right|center|none|border|upright)\s*\|
    )?
    (?:miniatur|mini|thumb|thumbnail|rahmenlos|framed?)\s*
    (?:\|(?:hochkant(?:=[0-9.]+)?|upright(?:=[0-9.]+)?|[0-9]+(?:x[0-9]+)?px|left|right|links|rechts|none|center|border))*\|?
    """
)
MEDIA_OPTION_RE = re.compile(
    r"(?ix)(?:hochkant(?:=[0-9.]+)?|upright(?:=[0-9.]+)?|[0-9]+(?:x[0-9]+)?px|15px|left|right|links|rechts|none|center|border)\|"
)
PLACEMENT_WORD_RE = re.compile(r"(?i)\b(?:center|left|right|links|rechts)(?=[A-ZÄÖÜ])")
ALT_TEXT_RE = re.compile(r"(?i)\|?\s*alternativtext\s*=\s*[^|\n]{0,1000}\|")
WIKI_LINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
MALFORMED_WIKI_LINK_RE = re.compile(r"\[\[([^\[\]]+)")
SKIP_TITLE_RE = re.compile(r"(?i)^(?:liste|lista)\b")
INFOBOX_JUNK_RE = re.compile(
    r"(?i)(?:\bDate\s+Namme?:|\bName:.*\bFlächi|\bZueflüss:|\bAbflüss:|"
    r"\bHöchi\s+(?:yber|über|ü\.M|NN)|\bBFS-Nr\.|Bsunderheite:|"
    r"Grösseri\s+\w+\s+am\s+Ufer|Charte\s+Dr\b|maximal[ei]\s+(?:Diefi|Tiefi))"
)
LIKELY_SENTENCE_START_RE = re.compile(
    r"""(?ix)
    \b(?:D|Dr|De|Ds|Di|Die|Der|Das|S|E|En|Es|Im|Am|Bi|Uf|Als)\s+
    [A-ZÄÖÜ][^.!?]{0,140}?
    \b(?:isch|is|het|hät|sind|si|lit|litt|liit|gilt|wird|wärd|cha|gits|gibt)\b
    """
)
LEADING_JUNK_RE = re.compile(
    # Word-bounded: unanchored, "charte" matched inside "Wältcharte" and made
    # the cleaner discard whole lead paragraphs as if they were image captions.
    r"(?i)\b(?:logo|wappe|laptop|blue gene|tablet pc|pdp-7|date\s+namme?:|lutwert|schrift mit name|charte|stimmig|blick)\b"
)

# A junk prefix is a caption fragment: short, and with no sentence-ending
# punctuation. Anything longer or containing a full stop is real prose.
MAX_JUNK_PREFIX_CHARS = 120

DROP_LINE_RE = re.compile(
    r"""(?ix)
    ^\s*
    (?:
        (?:links|rechts|left|right|center|none)?\|?\s*(?:mini|thumb|thumbnail|rahmenlos|framed?)\b.* |
        (?:mini|thumb|thumbnail|rahmenlos|framed?)\|.* |
        (?:Datei|File|Image|Bild)\s*:.* |
        (?:Kategorie|Category)\s*:.* |
        \[\[(?:Kategorie|Category)\s*:.* |
        __[^_]+__ |
        \{\{.*\}\} |
        \|.* |
        !.*
    )
    \s*$
    """
)

COMMON_HEADING_RE = re.compile(
    r"""(?ix)
    ^\s*
    (?:
        geograf(?:ie|ii|y)|gschicht|geschichte|literatur|weblinks?|webglingg|
        einzelnachwiis|nachwiis|quälle|quelle|referänze|referenze|
        läbe|werch|werke|persone|söh?nschtiges|sonschtigs?|siihe?\s+au
    )
    \s*$
    """
)


def strip_wiki_links(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        inner = match.group(1)
        if re.match(r"(?i)^(?:Datei|File|Image|Bild|Kategorie|Category):", inner):
            return " "
        if "|" in inner:
            return inner.rsplit("|", 1)[-1]
        if "}" in inner:
            return inner.rsplit("}", 1)[-1]
        return inner

    text = WIKI_LINK_RE.sub(repl, text)
    text = MALFORMED_WIKI_LINK_RE.sub(lambda m: m.group(1).rsplit("|", 1)[-1], text)
    text = text.replace("]]", " ")
    return text


def strip_media_markup(text: str) -> str:
    text = ALT_TEXT_RE.sub(MEDIA_MARK, text)
    text = re.sub(
        r"\[\[(?:Datei|File|Image|Bild):[^\]]+\]\]",
        MEDIA_MARK,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?i)(?:Datei|File|Image|Bild)\s*:[^.\n]{0,180}",
        MEDIA_MARK,
        text,
    )
    text = MEDIA_PREFIX_RE.sub(MEDIA_MARK, text)
    text = MEDIA_OPTION_RE.sub(MEDIA_MARK, text)
    text = PLACEMENT_WORD_RE.sub(" ", text)
    text = re.sub(r"(?i)(?:miniatur|mini|thumb|thumbnail|rahmenlos|framed?)\|?", MEDIA_MARK, text)
    return text


def clean_symbol_remnants(text: str) -> str:
    text = text.replace('\\"', '"')
    text = re.sub(r"(?i)\balternativtext\s*=\s*", " ", text)
    text = re.sub(r"\[\s*\]", " ", text)
    text = re.sub(r"(?<!\w)//(?!\w)", " ", text)
    for _ in range(3):
        text = re.sub(r"\(\s*(?:,\s*)+", "(", text)
        text = re.sub(r"(?:,\s*){2,}", ", ", text)
        text = re.sub(r",\s*\)", ")", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        text = re.sub(r"\s+([,;:.!?])", r"\1", text)
        text = re.sub(
            r"\b(stimmhafts|stimmloses|emphatisch(?:es)?)\s*:\s*",
            r"\1 ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\bFür\s*:\s*", "", text)
        text = text.replace("|", " ")
        text = re.sub(r"\(\s*\)", " ", text)
        text = re.sub(r"\(\s*;\s*\)", " ", text)
    return text


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\\n", "\n")
    text = text.translate(str.maketrans({
        "’": "'",
        "‘": "'",
        "‚": "'",
        "‛": "'",
        "´": "'",
        "`": "'",
        "–": "-",
        "—": "-",
        "−": "-",
    }))
    return text


def remove_media_caption_fragments(text: str, title: str | None = None) -> str:
    if MEDIA_MARK.strip() not in text:
        return text

    title_tokens = []
    if title:
        title_tokens = [tok for tok in re.split(r"\W+", title) if len(tok) >= 4]

    pieces = text.split(MEDIA_MARK)
    cleaned = [pieces[0]]
    for piece in pieces[1:]:
        title_positions = []
        window = piece[:300]
        for token in title_tokens:
            match = re.search(rf"\b{re.escape(token)}\b", window, flags=re.IGNORECASE)
            if match:
                title_positions.append(match.start())
        if title_positions:
            cleaned.append(piece[min(title_positions):])
            continue

        match = LIKELY_SENTENCE_START_RE.search(window)
        prefix = window[:match.start()] if match else ""
        if match and (match.start() <= 40 or LEADING_JUNK_RE.search(prefix)):
            cleaned.append(piece[match.start():])
        else:
            cleaned.append(piece)
    return " ".join(cleaned)


def strip_leading_junk_prefix(text: str) -> str:
    window = text[:500]
    match = LIKELY_SENTENCE_START_RE.search(window)
    if not match or match.start() == 0:
        return text
    prefix = text[:match.start()]
    # Guard against eating the article's lead paragraph: LIKELY_SENTENCE_START_RE
    # cannot match across a "." (e.g. "vo griech. geografia"), so the first match
    # can land far into the article and make real prose look like a prefix.
    if len(prefix) > MAX_JUNK_PREFIX_CHARS or re.search(r"[.!?]", prefix):
        return text
    if len(prefix) <= 45 or LEADING_JUNK_RE.search(prefix):
        return text[match.start():].lstrip()
    return text


def is_low_quality_record(title: str, text: str) -> bool:
    if SKIP_TITLE_RE.match(title):
        return True
    if INFOBOX_JUNK_RE.search(text[:800]):
        return True
    words = text.split()
    if len(words) < 20:
        return True
    if text.count("°") >= 12:
        return True
    digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    if digit_ratio > 0.18:
        return True
    return False


def clean_text(text: str, title: str | None = None) -> str:
    text = normalize_text(text)
    text = MAGIC_WORD_RE.sub(" ", text)
    text = re.sub(r"<ref\b[^>]*>.*?</ref>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<ref\b[^/]*/\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<ref\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[https?:/\s*/[^\]]+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"https?:/\s*/\S+", "", text, flags=re.IGNORECASE)
    text = strip_media_markup(text)
    text = strip_wiki_links(text)
    text = remove_media_caption_fragments(text, title)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = text.replace("]]>", " ")
    text = NON_LATIN_SCRIPT_RE.sub(" ", text)
    text = clean_symbol_remnants(text)
    text = re.sub(r"(?i)[0-9]+(?:x[0-9]+)?px", " ", text)

    lines = []
    in_template = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if in_template:
            if "}}" in line:
                in_template = False
            continue
        if line.startswith("}}"):
            continue
        if line.startswith("{{") and "}}" not in line:
            in_template = True
            continue
        if not line:
            lines.append("")
            continue
        if DROP_LINE_RE.match(line):
            continue
        line = strip_media_markup(strip_wiki_links(line))
        line = remove_media_caption_fragments(line, title)
        line = re.sub(r"[ \t]{2,}", " ", line).strip()
        if not line or COMMON_HEADING_RE.match(line):
            continue
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\b([A-Za-zÄÖÜäöüß]{1,4})(?:\s+\1\b){2,}", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"(?i)\|\s*(?:time\s*=\s*[\d:]+|left|right|links|rechts)\b", " ", text)
    text = re.sub(r"[:;,.!?]{2,}", lambda m: m.group(0)[0], text)
    text = text.replace(MEDIA_MARK.strip(), " ")
    text = clean_symbol_remnants(text)
    # Keep blank-line paragraph breaks: the translation chunker splits on them,
    # and flattening every newline to a space forced whole articles (up to
    # ~16k words) into a single API request. Single newlines inside a paragraph
    # still collapse to spaces.
    _PARA = "\x00PARA\x00"
    text = re.sub(r"[ \t]*\n(?:[ \t]*\n)+[ \t]*", _PARA, text)
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    text = text.replace(_PARA, "\n\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = strip_leading_junk_prefix(text)
    return text.strip()


def article_url(title: str) -> str:
    return "https://als.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="/_:")


def iter_records():
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = clean_text(row["text"], row["title"])
            if len(text) < MIN_CHARS:
                continue
            if is_low_quality_record(row["title"], text):
                continue
            # als_articles.jsonl carries the human Wikipedia category tag:
            # dialect_before_mapping is the fine dialect name ("Baseldytsch"),
            # dialect is its canton code. "ch" means "no category tag".
            fine = row.get("dialect_before_mapping")
            has_tag = bool(fine) and fine != "ch"
            yield {
                "id": row["id"],
                "source": "als.wikipedia",
                "source_url": article_url(row["title"]),
                "title": row["title"],
                "source_language": "gsw",
                "wiki_language": "als",
                "target_language": "de",
                "text": text,
                "dialect": fine if has_tag else None,
                "dialect_canton": row.get("dialect", "ch"),
                "dialect_label_source": "category" if has_tag else None,
                "char_count": len(text),
                "word_count": len(text.split()),
                "translation": None,
            }


def main() -> None:
    if not INPUT_FILE.exists():
        raise SystemExit(f"Input file not found: {INPUT_FILE}")

    total = written = 0
    dialects: Counter[str | None] = Counter()
    sources: Counter[str | None] = Counter()
    with open(INPUT_FILE, encoding="utf-8") as f:
        total = sum(1 for _ in f)

    HF_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out, \
         open(HF_DATA_FILE, "w", encoding="utf-8") as hf_out:
        for rec in iter_records():
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            hf_rec = {
                "text": rec["text"],
                "dialect": rec["dialect"] or "",
                "dialect_canton": rec["dialect_canton"],
            }
            hf_out.write(json.dumps(hf_rec, ensure_ascii=False) + "\n")
            written += 1
            dialects[rec["dialect"]] += 1
            sources[rec["dialect_label_source"]] += 1

    manifest = {
        "input": str(INPUT_FILE),
        "output": str(OUTPUT_FILE),
        "rows_input": total,
        "rows_written": written,
        "rows_dropped_after_cleaning_and_quality_filters": total - written,
        "min_chars": MIN_CHARS,
        "dialect_distribution": dict(dialects.most_common()),
        "dialect_label_source_distribution": dict(sources.most_common()),
        "notes": [
            "No machine translation has been run.",
            "text is the Alemannic/Swiss German source text to translate to target_language=de.",
            "translation is intentionally null for downstream Gemini output.",
            "Rows can be dropped when they become too short or look like list/table/infobox content after cleaning.",
        ],
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    HF_README_FILE.write_text(dataset_card(written, dialects, sources), encoding="utf-8")

    print(f"Wrote {written:,}/{total:,} rows -> {OUTPUT_FILE}")
    print(f"Wrote Hugging Face export -> {HF_DATA_FILE}")
    print(f"Manifest -> {MANIFEST_FILE}")


def dataset_card(
    rows_written: int,
    dialects: Counter[str | None],
    sources: Counter[str | None],
) -> str:
    known_dialects = rows_written - dialects.get(None, 0)
    category_tagged = sources.get("category", 0)
    unknown = sources.get(None, 0)
    dialect_lines = "\n".join(
        f"- `{dialect or '<empty>'}`: {count}"
        for dialect, count in dialects.most_common()
    )
    source_lines = "\n".join(
        f"- `{source or '<empty>'}`: {count}"
        for source, count in sources.most_common()
    )
    return f"""---
license: cc-by-sa-4.0
language:
- gsw
- de
task_categories:
- translation
pretty_name: Alemannic Wikipedia Clean Translation Corpus
---

# Alemannic Wikipedia Clean Translation Corpus

This private dataset contains cleaned source text from Alemannic Wikipedia for later translation into High German. No machine translation output is included.

## Columns

- `text`: cleaned Alemannic/Swiss German source article text.
- `dialect`: dialect label when known. This is an empty string when no dialect label is available.
- `dialect_canton`: two-letter canton code for the dialect, or `ch` when untagged.

## Dialect Labeling

Dialect labels come from a single source: the human-applied Alemannic Wikipedia
dialect categories.

Author-based inference (guessing an article's dialect from its main contributing
author) was evaluated and **dropped** — it agreed with the human category tags
only 25-33% of the time, and its confidence score did not predict correctness.

Rows without a category tag have an empty `dialect` field; those are classified
by the translation model at translation time instead.

## Counts

- Rows: {rows_written}
- Rows with known dialect: {known_dialects}
- Direct category labels: {category_tagged}
- Unknown dialect: {unknown}

## Dialect Distribution

{dialect_lines}

## Internal Label Source Distribution

This source field is not included as a dataset column, but was used to build the export:

{source_lines}

## Intended Use

This corpus is intended as input for Alemannic/Swiss German to High German translation workflows. The `text` column should be translated into High German; the other columns are metadata.

## Source and License

Source text comes from Alemannic Wikipedia. Reuse should comply with Wikipedia/Wikimedia licensing and attribution requirements.
"""


if __name__ == "__main__":
    main()
