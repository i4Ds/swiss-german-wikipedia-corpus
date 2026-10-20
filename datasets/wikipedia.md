---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
  - split: swiss_german
    path: data/swiss_german-*
dataset_info:
  features:
  - name: sentence_de
    dtype: string
  - name: sentence_ch
    dtype: string
  - name: dialect
    dtype: string
  - name: dialect_fine
    dtype: string
  - name: confidence
    dtype: int64
  - name: article_id
    dtype: string
  - name: segment_index
    dtype: int64
  - name: title
    dtype: string
  - name: source_url
    dtype: string
  - name: written_out_num_sentence_ch
    dtype: string
  - name: numbers
    list:
    - name: number
      dtype: string
    - name: type
      dtype: string
    - name: subtype
      dtype: string
    - name: value_json
      dtype: string
    - name: start
      dtype: int64
    - name: end
      dtype: int64
    - name: spoken_form
      dtype: string
    - name: annotation_json
      dtype: string
  - name: number_dialect
    dtype: string
  - name: number_dialect_source
    dtype: string
  - name: number_processing_status
    dtype: string
  - name: number_processing_error
    dtype: string
  splits:
  - name: train
    num_examples: 215889
    num_bytes: 143344437
  - name: validation
    num_examples: 12736
    num_bytes: 8461565
  - name: test
    num_examples: 12548
    num_bytes: 8120466
  - name: swiss_german
    num_examples: 86272
    num_bytes: 56125300
  download_size: 61618921
  dataset_size: 216051768
pretty_name: Wikipedia German–Swiss German Translation and Number Annotations
language:
- de
- gsw
task_categories:
- translation
---
# Wikipedia German–Swiss German sentence pairs with number annotations

A parallel text resource derived from the Alemannic Wikipedia, with German
translations of cleaned Swiss German source segments. It contains **327,445
sentence pairs**, original article metadata, dialect labels and explicit numeric
annotations, together with written-out Swiss German number targets for translator
training and subsequent speech preparation. The German translations and regional
dialect assignments are automatically generated; they are not all human gold labels.

## Splits and source metadata

| Split | Rows | Intended use |
|---|---:|---|
| `train` | 215,889 | Training |
| `validation` | 12,736 | Model selection |
| `test` | 12,548 | Final evaluation |
| `swiss_german` | 86,272 | Separate supplementary source subset |

Original sentence pairs, IDs, row order and split assignments are preserved.
`sentence_de` is the German side; `sentence_ch` is the cleaned Swiss German side.
`article_id`, `segment_index`, `title` and `source_url` retain article provenance.
`dialect` is legacy regional/canton metadata; `dialect_fine` preserves the fine
regional label, and `confidence` records the existing dialect confidence tier.
Legacy coarse labels can be generic even when a fine region is present. The
`swiss_german` subset also contains some supported regional metadata, so its
name should not be interpreted as a guarantee that every row has only `ch`.

## Number annotations and five-dialect verbalisation

The October 2026 update reuses the saved semantic number labels. No new LLM
inference or translation was performed for that update. Basel (`ch_bs`), St. Gallen
(`ch_sg`), Zürich (`ch_zh`), Luzern (`ch_lu`) and Bern (`ch_be`) converters are
supported. A supported `dialect_fine` label takes precedence, followed by a
supported `dialect` code. **Every other or generic region uses St. Gallen number
forms.** Original labels remain unchanged; per-row provenance records the actual
converter, including fallback use.

| Column | Meaning |
|---|---|
| `written_out_num_sentence_ch` | Swiss German source with annotated number spans verbalised; null on failure |
| `numbers` | Native list of numeric annotation objects |
| `number_dialect` | Converter used for a numeric row |
| `number_dialect_source` | `dialect_fine`, `dialect`, or explicit `fallback` |
| `number_processing_status` | `no_numbers`, `converted`, `inference_error` or `conversion_error` |
| `number_processing_error` | Failure reason when applicable |

Each annotation includes `number`, `type`, `subtype`, `start`, `end`, `spoken_form`,
`value_json` and `annotation_json`. Offsets refer to the original `sentence_ch`,
with start inclusive and end exclusive. `value_json` preserves heterogeneous
numeric values; `annotation_json` preserves all annotation fields, including
grammatical metadata. Both can be decoded using `json.loads`; the surrounding
`numbers` column is a native list of objects.

```python
import json
from datasets import load_dataset

data = load_dataset("i4ds/translator_data_wikipedia", split="train", token=True)
usable = data.filter(lambda row: row["number_processing_status"] in {"converted", "no_numbers"})
numeric = next(row for row in usable if row["numbers"])
value = json.loads(numeric["numbers"][0]["value_json"])
annotation = json.loads(numeric["numbers"][0]["annotation_json"])
```

## Validation and limitations

The number release validated **123,430** converted numeric rows by reconstructing
their written targets from exported spans and spoken forms. **2,069** rows have
inference or conversion failures: their source records remain present with a null
written target and an explicit status. Rows without detected decimal digits have
unchanged written text and `numbers=[]`; unavailable inference has `numbers=null`.
Parsed annotations remain present when conversion fails.

Number selection retains the existing Unicode decimal-digit (`\d`) detection;
superscripts are not newly expanded. Existing deterministic annotation repairs
are applied before conversion. Annotation confidence is not a guarantee of
translation or verbalisation correctness; errors can remain in machine-labelled
text. See [release manifest](https://huggingface.co/datasets/i4ds/translator_data_wikipedia/blob/d0491d0112403763d272b193a7c74534bfcec5f4/number_preparation/release_manifest.json) for converter
revision, counts and data checksums.
