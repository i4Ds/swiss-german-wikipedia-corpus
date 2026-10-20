#!/bin/bash
# Gemini 3.5 Flash dialect/translation experiment on the 300-article eval set.
#
# MUST run on the LOGIN node: compute nodes cannot reach openrouter.ai
# (neither directly nor via the squid proxy — verified 2026-07-16).
# The run is API-bound (~0% CPU), so it does not load the login node.
#
# Needs OPENROUTER_API_KEY in the environment or a .env next to this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-${WIKI_DATA_ROOT:-${DATA_ROOT:-${HOME}/.local/share/swiss-german-wikipedia-corpus}}/results}"
EVAL_DIR="$OUT_DIR/gemini_eval"

PYTHON_BIN="${PYTHON:-python3}"

cd "$SCRIPT_DIR"
"$PYTHON_BIN" -u run_openrouter_wikipedia_translate.py \
    --input_jsonl      "$EVAL_DIR/eval_set_300.jsonl" \
    --output_jsonl     "$EVAL_DIR/translated_300.jsonl" \
    --checkpoint_jsonl "$EVAL_DIR/checkpoint_300.jsonl" \
    --model            google/gemini-3.5-flash \
    --reasoning_effort minimal \
    --max_chunks_per_article 2 \
    "$@"

"$PYTHON_BIN" analyze_dialect_eval.py "$EVAL_DIR/translated_300.jsonl"
