# Optional author dialect experiments

The published translation input uses Wikipedia category labels and subsequent
model-assigned regional labels. Author-inferred labels are experimental: earlier
evaluations showed low agreement with category labels, so they are excluded
from translation input.

All scripts use `corpus_paths.py`: set `WIKI_DATA_ROOT` and optionally `OUT_DIR`.
The history dump defaults to `$WIKI_DATA_ROOT/raw/alswiki-latest-stub-meta-history.xml.gz`;
set `WIKI_HISTORY_DUMP` to use another location.

```bash
RUN_AUTHOR_EXPERIMENTS=1 bash run_pipeline.sh
```

The individual scripts parse author contributions, optionally scrape explicit
self-declarations, infer author dialects and join them to article text. They
write their experimental outputs to the results directory. The clean exporter
reads extracted articles directly and keeps only their human category tags.
