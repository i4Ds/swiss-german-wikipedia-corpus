#!/usr/bin/env python3
"""Translate the cleaned Alemannic Wikipedia corpus to Standard High German.

Reads clean_translation_corpus.jsonl produced by
author_dialect/export_clean_translation_corpus.py. For each article the text is
sent in chunks to Gemini via OpenRouter for final cleaning and translation.

Output JSONL record per article:
  {
    "id":                    "<page_id>",
    "title":                 "<title>",
    "dialect":               "<alemannic_dialect_name or null>",
    "dialect_confidence":    0.0,
    "gemini_dialect_analyses": [
      {"label": "Zürich", "confidence_level": 0-3}
    ],
    "segments": [
      {"gsw": "<cleaned Swiss German segment>", "de": "<HG translation>"}
    ]
  }

Resume: already-processed article IDs are skipped on re-run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_paths import ARTICLE_DUMP, HISTORY_DUMP, EXTRACTED_DIR, RESULTS_DIR
from typing import Any

from PROMPT import (
    PROMPT_WIKIPEDIA_CLEAN_TRANSLATE_DIALECT_V2 as PROMPT_WIKIPEDIA_CLEAN_TRANSLATE,
)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = str(RESULTS_DIR / "clean_translation_corpus.jsonl")
DEFAULT_OUTPUT_JSONL = RESULTS_DIR / "translated/wikipedia_translated.jsonl"
DEFAULT_CHECKPOINT = RESULTS_DIR / "translated/checkpoint.jsonl"

TRANSLATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "input":  {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["input", "output"],
                "additionalProperties": False,
            },
        },
        "dialect": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": [
                        "Zürich", "Luzern", "St. Gallen", "Bern",
                        "Basel", "Wallis", "Graubünden",
                        "Swiss German", "Not Swiss German",
                    ],
                },
                # NOTE: no enum here — Google's schema validator only accepts
                # enums on string types and rejects the whole schema otherwise.
                # The 0-3 range is enforced by the prompt.
                "confidence_level": {"type": "integer"},
            },
            "required": ["label", "confidence_level"],
            "additionalProperties": False,
        },
    },
    "required": ["translations", "dialect"],
    "additionalProperties": False,
}

# ── pre-cleaning regex ────────────────────────────────────────────────────────

# Patterns to strip before sending to the LLM
_WIKI_NOISE = re.compile(
    r"(?:"
    # image/thumb markup: rechts|mini|..., links|mini|..., mini|300px|...,
    # border|100px|none|..., thumb|..., upright|...
    r"(?:rechts|links|none|border|upright)\s*\|[^\n]*"
    r"|(?:mini|thumb)\s*\|[^\n]*"
    r"|\d+px\s*\|[^\n]*"
    r")"
)

_BARE_URL = re.compile(r"https?://\S+")
_REF_MARKER = re.compile(r"\[\d+\]")
_PIPE_LINE = re.compile(r"^\s*\|.*$", re.MULTILINE)
# Remove words/sequences from non-Latin scripts (Greek, Cyrillic, Arabic, Devanagari, CJK, etc.)
_NON_LATIN_SCRIPT = re.compile(
    r"[\u0370-\u03FF\u0400-\u04FF\u0600-\u06FF\u0900-\u097F"
    r"\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]+"
)


def pre_clean(text: str) -> str:
    """Remove the most obvious wiki artifacts before LLM processing."""
    text = _WIKI_NOISE.sub("", text)
    text = _BARE_URL.sub("", text)
    text = _REF_MARKER.sub("", text)
    text = _PIPE_LINE.sub("", text)
    text = _NON_LATIN_SCRIPT.sub("", text)
    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── chunk splitter ─────────────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_long_paragraph(para: str, max_words: int) -> list[str]:
    """Break a single over-long paragraph on sentence boundaries.

    The corpus stores each article as one paragraph (no blank lines), so without
    this the paragraph loop below never splits and whole articles — up to
    ~16k words — go out in one request, risking truncated output.
    """
    pieces: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in _SENTENCE_END.split(para):
        wc = len(sentence.split())
        if current and current_words + wc > max_words:
            pieces.append(" ".join(current))
            current = [sentence]
            current_words = wc
        else:
            current.append(sentence)
            current_words += wc
    if current:
        pieces.append(" ".join(current))
    return pieces


MAX_CHUNK_WORDS = 6382  # validated: 100% coverage, tail intact, 30,471/65,536 out tokens


def split_into_chunks(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Split article text into chunks of at most max_words words.

    Splits on paragraph boundaries where they exist, and falls back to sentence
    boundaries inside any paragraph that is itself longer than max_words.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if len(para.split()) > max_words:
            units.extend(_split_long_paragraph(para, max_words))
        else:
            units.append(para)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for unit in units:
        wc = len(unit.split())
        if current and current_words + wc > max_words:
            chunks.append("\n\n".join(current))
            current = [unit]
            current_words = wc
        else:
            current.append(unit)
            current_words += wc

    if current:
        chunks.append("\n\n".join(current))
    return chunks


# ── API client ────────────────────────────────────────────────────────────────

def _build_client():
    import httpx
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(SCRIPT_DIR / ".env", override=False)
    load_dotenv(SCRIPT_DIR.parent / ".env", override=False)
    load_dotenv(override=False)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    # Explicitly disable proxy: openrouter.ai must be reached directly.
    # httpx otherwise picks up http_proxy/https_proxy env vars which may block the request.
    http_client = httpx.Client(proxy=None)
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, http_client=http_client)


def _validate_response(parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False
    translations = parsed.get("translations")
    if not isinstance(translations, list):
        return False
    if not isinstance(parsed.get("dialect"), dict):
        return False
    for item in translations:
        if not isinstance(item, dict):
            return False
        if "input" not in item or "output" not in item:
            return False
        if not isinstance(item["input"], str) or not isinstance(item["output"], str):
            return False
    return True


def _strip_code_fence(content: str) -> str:
    """Unwrap ```json ... ``` fences some models emit around JSON.

    Strict schema mode normally prevents this, but a fenced response would
    otherwise fail json.loads and burn a full paid retry.
    """
    s = content.strip()
    if not s.startswith("```"):
        return s
    s = s.split("\n", 1)[1] if "\n" in s else s[3:]      # drop ```/```json line
    if s.rstrip().endswith("```"):
        s = s.rstrip()[:-3]
    return s.strip()


def _normalize_confidence(parsed: dict[str, Any]) -> None:
    """Force confidence_level back onto the prompt's 0-3 scale, in place.

    Google's schema validator rejects enums on integer properties, so the
    0-3 range cannot be enforced by the schema and the model occasionally
    emits something else (observed: 23). The label->confidence mapping is
    fully determined by PART 6, so rederive it rather than guessing.
    """
    dialect = parsed.get("dialect")
    if not isinstance(dialect, dict):
        return
    conf = dialect.get("confidence_level")
    label = dialect.get("label")
    if label == "Not Swiss German":
        dialect["confidence_level"] = 0            # prompt rule 9
    elif label == "Swiss German":
        dialect["confidence_level"] = 1            # prompt rule 8
    elif conf not in (2, 3):
        # A regional dialect must be 2 or 3 (rule 10). 2 vs 3 is the model's
        # own evidence signal, so keep it when valid; otherwise take the
        # weaker level rather than overstate the evidence.
        dialect["confidence_level"] = 2


def _call_api(
    client: OpenAI,
    model: str,
    chunk: str,
    max_retries: int,
    retry_sleep: float,
    reasoning_effort: str,
    service_tier: str = "default",
) -> dict[str, Any]:
    for attempt in range(1, max_retries + 1):
        try:
            extra_body: dict[str, Any] = {}
            if reasoning_effort != "none":
                extra_body["reasoning"] = {"effort": reasoning_effort}
            if service_tier != "default":
                # "flex" = ~50% cheaper, higher latency / lower availability.
                extra_body["service_tier"] = service_tier

            response: Any = client.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": PROMPT_WIKIPEDIA_CLEAN_TRANSLATE.strip()},
                    {"role": "user", "content": chunk},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "translation_schema",
                        "strict": True,
                        "schema": TRANSLATION_SCHEMA,
                    },
                },
                extra_headers={
                    "HTTP-Referer": "https://local.als-wikipedia-translate",
                    "X-Title": "als_wikipedia_translate",
                },
                extra_body=extra_body or None,
            )
            # On a provider-side error (flex capacity, rate limit) OpenRouter
            # returns an error payload with no `choices`. Surface that message
            # instead of letting it masquerade as a TypeError.
            choices = getattr(response, "choices", None)
            if not choices:
                err = getattr(response, "error", None)
                raise ValueError(f"No choices returned: {err or response}")
            content = choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty response from model.")
            parsed = json.loads(_strip_code_fence(content))
            if not _validate_response(parsed):
                raise ValueError("Response JSON does not match schema.")
            _normalize_confidence(parsed)
            usage = getattr(response, "usage", None)
            return {
                "ok": True,
                "parsed": parsed,
                "error": None,
                # OpenRouter reports the real charge per call; sum it to get the
                # actual spend rather than an estimate from list prices.
                "cost": float(getattr(usage, "cost", 0.0) or 0.0),
                "in_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "out_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            }
        except Exception as exc:
            if attempt >= max_retries:
                return {"ok": False, "parsed": None, "error": str(exc),
                        "cost": 0.0, "in_tokens": 0, "out_tokens": 0}
            # Exponential backoff: flex errors are transient capacity blips and
            # a fixed short sleep just burns the retry budget during one.
            time.sleep(retry_sleep * (2 ** (attempt - 1)))
    return {"ok": False, "parsed": None, "error": "Retry loop exhausted.",
            "cost": 0.0, "in_tokens": 0, "out_tokens": 0}


# ── main ──────────────────────────────────────────────────────────────────────

def run(
    input_jsonl: Path,
    output_jsonl: Path,
    checkpoint_jsonl: Path,
    model: str,
    max_retries: int,
    retry_sleep: float,
    reasoning_effort: str,
    resume: bool,
    max_articles: int | None,
    max_chunks_per_article: int | None = None,
    service_tier: str = "default",
    concurrency: int = 1,
    max_chunk_words: int = MAX_CHUNK_WORDS,
) -> None:
    client = _build_client()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Load already-processed IDs for resume
    done_ids: set[str] = set()
    if resume and checkpoint_jsonl.exists():
        with checkpoint_jsonl.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        if done_ids:
            print(f"Resuming: {len(done_ids)} articles already processed.")

    # Build the work list up front, skipping resumed IDs and de-duplicating the
    # input itself, so an article can never be billed twice in one run.
    todo: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    dupes = 0
    with input_jsonl.open(encoding="utf-8") as fin:
        for line in fin:
            if max_articles is not None and len(todo) >= max_articles:
                break
            article = json.loads(line)
            art_id = str(article["id"])
            if art_id in done_ids:
                continue
            if art_id in seen_ids:
                dupes += 1
                continue
            seen_ids.add(art_id)
            article["id"] = art_id
            todo.append(article)

    if dupes:
        print(f"Skipped {dupes} duplicate id(s) in the input file.")
    print(f"To process: {len(todo)} articles with concurrency={concurrency}.")

    counters = {"total": 0, "failed_chunks": 0, "chunks": 0, "failed_articles": 0,
                "cost": 0.0, "in_tokens": 0, "out_tokens": 0}
    failed_ids: list[str] = []
    io_lock = threading.Lock()
    start = time.time()

    def process(article: dict[str, Any]) -> None:
        art_id = article["id"]
        title = article.get("title", "")
        dialect = article.get("dialect")

        chunks = split_into_chunks(pre_clean(article.get("text", "")), max_chunk_words)
        if max_chunks_per_article is not None:
            chunks = chunks[:max_chunks_per_article]

        segments: list[dict[str, str]] = []
        gemini_dialect_analyses: list[dict[str, Any]] = []
        article_ok = True
        art_cost = 0.0
        local_failed = 0

        for chunk_idx, chunk in enumerate(chunks):
            result = _call_api(
                client, model, chunk, max_retries, retry_sleep, reasoning_effort,
                service_tier
            )
            art_cost += result.get("cost", 0.0)
            if not result["ok"]:
                print(f"  [{title[:40]}] chunk {chunk_idx} FAILED: {result['error']}")
                local_failed += 1
                article_ok = False
                continue

            gemini_dialect_analyses.append(result["parsed"]["dialect"])
            for item in result["parsed"]["translations"]:
                gsw = item["input"].strip()
                de = item["output"].strip()
                if gsw and de and de not in ("None", "NA", ""):
                    segments.append({"gsw": gsw, "de": de})
                elif gsw and de == "None":
                    # Already HG — still useful, keep with de==gsw
                    segments.append({"gsw": gsw, "de": gsw})

            with io_lock:
                counters["in_tokens"] += result.get("in_tokens", 0)
                counters["out_tokens"] += result.get("out_tokens", 0)

        record = {
            "id": art_id,
            "title": title,
            "source_url": article.get("source_url"),
            "dialect": dialect,
            "dialect_canton": article.get("dialect_canton"),
            "dialect_label_source": article.get("dialect_label_source"),
            "dialect_confidence": article.get("dialect_confidence", 0.0),
            "dialect_declaration_only": article.get("dialect_declaration_only", False),
            "eval_tier": article.get("eval_tier"),
            "author": article.get("author"),
            "gemini_dialect_analyses": gemini_dialect_analyses,
            "segments": segments,
            "api_cost_usd": round(art_cost, 6),
            "ok": article_ok,
        }

        # Single lock around both writes so output and checkpoint stay in step:
        # a crash can then only lose work, never silently mark it done.
        with io_lock:
            # Write output and checkpoint only for fully-successful articles.
            # Checkpointing a failure would make a resume skip it forever
            # (silent data loss); writing its partial record without
            # checkpointing would duplicate the id when the resume retries it.
            if article_ok:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                ckpt.write(json.dumps({"id": art_id}) + "\n")
                ckpt.flush()
            else:
                counters["failed_articles"] += 1
                failed_ids.append(art_id)
            counters["total"] += 1
            counters["failed_chunks"] += local_failed
            counters["chunks"] += len(chunks)
            counters["cost"] += art_cost
            n = counters["total"]
            if n % 25 == 0 or n == len(todo):
                el = time.time() - start
                rate = n / el * 3600
                print(f"  [{n}/{len(todo)}] {el/60:.1f} min elapsed | "
                      f"{rate:.0f} art/h | ${counters['cost']:.3f} spent", flush=True)

    with (
        output_jsonl.open("a", encoding="utf-8") as fout,
        checkpoint_jsonl.open("a", encoding="utf-8") as ckpt,
    ):
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(process, a) for a in todo]
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc is not None:
                    print(f"  WORKER ERROR: {exc}")

    elapsed = time.time() - start
    c = counters
    print(f"\nDone. {c['total']} articles / {c['chunks']} chunks, "
          f"{c['failed_chunks']} failed chunks, {c['failed_articles']} failed articles.")
    if failed_ids:
        print(f"NOT checkpointed (re-run to retry): {len(failed_ids)} ids -> "
              f"{', '.join(failed_ids[:10])}{' ...' if len(failed_ids) > 10 else ''}")
    print(f"Elapsed: {elapsed/60:.1f} min  ({c['total']/elapsed*3600:.0f} articles/h, "
          f"{c['chunks']/elapsed*3600:.0f} chunks/h)")
    print(f"Tokens: in={c['in_tokens']:,} out={c['out_tokens']:,}")
    print(f"COST: ${c['cost']:.4f} total, ${c['cost']/max(c['chunks'],1):.5f}/chunk, "
          f"${c['cost']/max(c['total'],1):.5f}/article")
    print(f"Output → {output_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and translate Swiss German Wikipedia articles to Standard High German."
    )
    parser.add_argument("--input_jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--output_jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--checkpoint_jsonl", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview"),
    )
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument(
        "--max_chunk_words",
        type=int,
        default=MAX_CHUNK_WORDS,
        help="Split a paragraph only when it exceeds this many words, and then "
             "only at a sentence boundary (. ! ?). Never splits mid-sentence.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel in-flight API requests. The job is network-bound, so "
             "8-16 is safe and cuts wall-clock time proportionally.",
    )
    parser.add_argument(
        "--service_tier",
        default="default",
        choices=["default", "flex", "priority"],
        help="OpenRouter service tier; 'flex' is ~50%% cheaper with higher latency.",
    )
    parser.add_argument(
        "--reasoning_effort",
        choices=["none", "minimal", "low", "medium", "high"],
        default=os.getenv("OPENROUTER_REASONING_EFFORT", "none"),
    )
    parser.add_argument(
        "--max_chunks_per_article", type=int, default=None,
        help="Only send the first N chunks of each article (cost cap for experiments).",
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Ignore checkpoint and reprocess all articles.",
    )
    parser.add_argument(
        "--max_articles", type=int, default=None,
        help="Stop after this many articles (useful for testing).",
    )
    args = parser.parse_args()

    run(
        input_jsonl=Path(args.input_jsonl),
        output_jsonl=Path(args.output_jsonl),
        checkpoint_jsonl=Path(args.checkpoint_jsonl),
        model=args.model,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        reasoning_effort=args.reasoning_effort,
        service_tier=args.service_tier,
        concurrency=args.concurrency,
        max_chunk_words=args.max_chunk_words,
        resume=not args.no_resume,
        max_articles=args.max_articles,
        max_chunks_per_article=args.max_chunks_per_article,
    )


if __name__ == "__main__":
    main()
