# Swiss German Wikipedia corpus

Standalone extraction, cleaning, translation, dialect analysis and export tools
for the Alemannic Wikipedia. This repository contains code, prompts, dataset
cards and release metadata. Corpus files are versioned on Hugging Face.

## Published resources

- [Wikipedia translator dataset](https://huggingface.co/datasets/i4ds/translator_data_wikipedia): 327,445 sentence pairs with number annotations and five-dialect number forms.
- [Clean translation corpus](https://huggingface.co/datasets/i4ds/alemannic-wikipedia-clean-translation-corpus): cleaned and aligned Swiss German / German text.
- Pinned revisions and SHA-256 checksums: [datasets/releases.json](datasets/releases.json).
- Dataset cards: [translator data](datasets/wikipedia.md), [source corpus](datasets/source-corpus.md).
- [Dialect distribution](datasets/statistics/wiki_dialect_distribution.csv) and [character counts](datasets/statistics/character_counts.csv).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export WIKI_DATA_ROOT=/path/to/wikipedia-data
```

Copy `.env.example` to `.env` and fill in credentials when using APIs. Load the
storage settings into your shell. Downloaded data and run outputs stay outside Git.

## Download the pinned published data

```bash
python scripts/download_release.py wikipedia --output-dir "$WIKI_DATA_ROOT/releases"
python scripts/download_release.py source-corpus --output-dir "$WIKI_DATA_ROOT/releases"
```

Access to private dataset releases uses `HF_TOKEN` or a cached Hugging Face login.
The script verifies available file checksums and never replaces different local data.

## Rebuild from Wikipedia dumps

Put `alswiki-latest-pages-articles.xml.bz2` in `$WIKI_DATA_ROOT/raw/`, or set
`WIKI_ARTICLE_DUMP` to an existing dump. Then run:

```bash
bash run_pipeline.sh
```

The wrapper fetches missing category metadata, extracts Swiss articles and exports
clean translation input. Existing category metadata is reused; set
`REFRESH_CATEGORIES=1` to refresh it. Extraction outputs go to
`$WIKI_DATA_ROOT/intermediate/extracted/`; clean input and manifests go to
`$WIKI_DATA_ROOT/results/`. `OUT_DIR` can override the results directory.

Author-based dialect inference is an optional experiment. It is excluded from
translation-input labels. Set `RUN_AUTHOR_EXPERIMENTS=1` and provide the history
dump through `WIKI_HISTORY_DUMP` to run it; see [author analysis](author_dialect/README.md).

## Translate, align and export

```bash
python OpenRouter_wikipedia_translate/run_openrouter_wikipedia_translate.py --help
python OpenRouter_wikipedia_translate/run_openrouter_wikipedia_translate.py
python OpenRouter_wikipedia_translate/scripts/flatten_segments_jsonl.py
python OpenRouter_wikipedia_translate/scripts/push_translation_corpus_to_hf.py --dry_run --out_dir /path/to/export-preview
```

Translation requires `OPENROUTER_API_KEY`. Model selection and generation options
are exposed by the runner. The original prompt is included in `PROMPT.py`.
Translation outputs and checkpoints default to `$WIKI_DATA_ROOT/results/translated/`.
The exporter stages data by default in dry-run mode; uploading is an explicit action.
Optional scheduler examples use the currently activated Python environment.

## Number preparation

Number annotation and verbalisation live in the companion
`swiss-german-number-preparation` repository. Its Wikipedia publisher preserves
original source text and adds `numbers`, written-number text and processing status.
It supports Basel, St. Gallen, Zürich, Luzern and Bern; other regions use the
explicit St. Gallen fallback. This repository does not duplicate those converters.

## Validation and provenance

```bash
python -m unittest discover -s tests -v
```

Tests cover dump extraction, non-Swiss filtering, source-label provenance,
paragraph preservation and aligned-segment reconstruction without API calls.
`provenance/extraction.json` identifies the original repository snapshot and file
checksums, including the working-tree fixes present at extraction. Source
attribution is recorded in [NOTICE.md](NOTICE.md).
