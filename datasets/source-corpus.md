---
license: cc-by-sa-4.0
language:
- gsw
- de
task_categories:
- translation
- text-generation
tags:
- swiss-german
- alemannic
- wikipedia
- parallel-corpus
- dialect
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
dataset_info:
  features:
  - name: de
    dtype: string
  - name: gsw
    dtype: string
  - name: dialect
    dtype: string
  - name: confidence
    dtype: int64
  - name: article_id
    dtype: string
  - name: segment_index
    dtype: int64
  - name: pair_id
    dtype: string
  - name: title
    dtype: string
  - name: source_url
    dtype: string
  - name: dialect_canton
    dtype: string
  - name: dialect_category
    dtype: string
  - name: dialect_category_source
    dtype: string
  - name: n_words_gsw
    dtype: int64
  - name: n_chars_gsw
    dtype: int64
  splits:
  - name: train
    num_bytes: 131295306
    num_examples: 327445
  download_size: 53155338
  dataset_size: 131295306
---
# Alemannic Wikipedia Clean Translation Corpus

A parallel **Swiss German ↔ Standard High German** corpus built from the
[Alemannic Wikipedia](https://als.wikipedia.org), with a regional dialect label
on every sentence pair.

- **327,445** sentence pairs
- **18,355** source articles
- **6,029,272** Swiss German words / **37,646,890** characters
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
| `pair_id` | `"{article_id}:{segment_index}"` — unique key for every row |
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

| Dialect | Sentence pairs | % | Articles | Words (gsw) | Chars (gsw) | Avg chars/pair |
|---|---:|---:|---:|---:|---:|---:|
| Basel | 113,936 | 34.8% | 6,418 | 2,256,932 | 13,998,277 | 123 |
| Swiss German | 86,272 | 26.3% | 5,191 | 1,499,560 | 9,471,740 | 110 |
| Zürich | 61,384 | 18.7% | 3,348 | 1,114,787 | 7,034,953 | 115 |
| Bern | 33,917 | 10.4% | 1,883 | 598,885 | 3,679,748 | 108 |
| St. Gallen | 14,337 | 4.4% | 664 | 252,747 | 1,561,716 | 109 |
| Luzern | 6,956 | 2.1% | 381 | 114,116 | 710,114 | 102 |
| Graubünden | 6,114 | 1.9% | 254 | 115,892 | 708,065 | 116 |
| Wallis | 4,529 | 1.4% | 216 | 76,353 | 482,277 | 106 |

## Confidence levels

| Level | Pairs | % | Meaning |
|---|---:|---:|---|
| 3 | 220,466 | 67.3% | Regional dialect — both identification methods agree |
| 2 | 20,707 | 6.3% | Regional dialect — one method, or methods disagree on region |
| 1 | 86,272 | 26.3% | Swiss German, but no region identifiable |
| 0 | 0 | 0.0% | Not Swiss German (excluded from this corpus) |

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
   Translating the full 25,287 articles cost
   **$205.34** in API spend
   (≈ $0.0081 per article),
   including the non-Swiss articles that were paid for and then filtered out.
5. **Filtering** — the Alemannic Wikipedia also contains **non-Swiss** Alemannic
   (Swabian, Badisch, Alsatian, Vorarlbergisch, non-Swiss Walser). Those
   articles are detected, labelled `Not Swiss German`, and **excluded** from
   this corpus (6,932 articles).

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
PROMPT_WIKIPEDIA_CLEAN_TRANSLATE_DIALECT_V2 = """
You are a Swiss German text cleaner, translator, and dialect analyst.

Your task has THREE main objectives:
1. CLEAN and segment the Wikipedia text.
2. TRANSLATE each cleaned Swiss German segment into Standard High German.
3. DETERMINE whether the text is Swiss German and, when possible, identify its regional dialect.

---

### PART 1 — WHAT TO CLEAN

Remove or skip the following:

1. Wikipedia image/media markup fragments:

   * Lines or parts of lines such as:
     mini|320px|..., rechts|mini|..., links|mini|...,
     thumb|..., 100px|..., none|..., border|...

2. Infobox or table parameter lines:

   * | KEY = VALUE
   * Standalone | characters or table fragments

3. Sentences or lines entirely in a foreign language, including Standard High
   German, French, Italian, Latin, or English, when no Swiss German is mixed in.

   * REMOVE:
     "Im Stadtzentrum gibt es viele Sehenswürdigkeiten."
     This is entirely Standard High German.

   * KEEP:
     "D Stadt Basel (frz. Bâle) isch..."
     A foreign-language label is embedded in Swiss German.

   * KEEP:
     "Ds Velo (Fahrrad) isch..."
     A Standard High German parenthetical explanation is embedded in Swiss German.

4. Bare URLs, reference markers such as [1] or [2], and citation stubs.

5. Lines containing only punctuation, numbers, or whitespace.

Additionally, note the following about abbreviations:

6. Do NOT expand abbreviations. Keep every abbreviation exactly in the form
   in which it appears in the Swiss German source text.

   * "z. B." stays "z. B."
   * "Frankfurt a. M." stays "Frankfurt a. M."
   * "12 km" stays "12 km"
   * FDP, SBB, USA, and similar acronyms stay unchanged.

   If the source text already writes something out in full, keep it written
   out. Never guess what an abbreviation stands for and never introduce an
   abbreviation that is not in the source.

---

### PART 2 — SEGMENTATION RULES

After cleaning, divide the text into translation units.

1. A segment ends at sentence-ending punctuation:
   `.`, `!`, or `?`

2. A blank line ends a segment.

3. A line break ends a segment if:

   * the line has sentence-ending punctuation, or
   * the running segment exceeds approximately 40 to 50 words.

4. Short lines without sentence-ending punctuation continue the current segment.

5. Do not merge text across paragraph boundaries.

6. Keep each segment below approximately 50 words.

7. Skip segments that are completely removed during cleaning.

---

### PART 3 — TRANSLATION RULES

1. Translate Swiss German vocabulary into natural Standard High German.

   Examples:

   * Velo → Fahrrad
   * Tram → Straßenbahn
   * Znüni → Pausensnack
   * isch → ist
   * het / hät → hat
   * chunt → kommt

2. Translate relative constructions naturally.

   Examples:

   * "wo i"
   * "woni"
   * "wo er"

   Do not translate these mechanically. Use a natural Standard High German
   relative clause.


3. Use "NA" if the segment cannot be understood or translated reliably.

4. Mirror the Swiss German segment's use of abbreviations exactly.

   * If the Swiss German segment writes something out in full, write it out
     in full in the translation too:
     "z Frankfurt am Main" → "in Frankfurt am Main"
   * If the Swiss German segment uses an abbreviation, keep an abbreviation
     in the translation too:
     "z Frankfurt a. M." → "in Frankfurt a. M."
     "z. B." → "z. B."   "12 km" → "12 km"

   Never expand an abbreviation that the source abbreviates, and never
   abbreviate something the source writes out.

### PART 4 — SWISS GERMAN DETERMINATION

First determine whether the cleaned text is Swiss German.

* If no usable text remains after cleaning, return an empty `"translations"`
  list, `"Not Swiss German"` as the label, and confidence level `0`.
* Only compare the general linguistic assessment and the explicit regional
  dialect rules after determining that the text is Swiss German.
* If the cleaned text is not Swiss German, return `"Not Swiss German"` as the
  label and confidence level `0`.

This Wikipedia covers all of Alemannic, not only Switzerland. It therefore
contains a substantial amount of text in Alemannic varieties spoken OUTSIDE
Switzerland:

* Swabian (Baden-Württemberg, e.g. Stuttgart region)
* Low and High Alemannic of Baden (e.g. Freiburg, Weil der Stadt)
* Alsatian (France)
* Vorarlbergisch (Austria) and Walser varieties of Italy (e.g. Pomatt)

These are Alemannic but NOT Swiss German. Label them `"Not Swiss German"`
with confidence level `0`, exactly like any other non-Swiss text.

IMPORTANT — do NOT translate text that is not Swiss German.
As soon as you determine that the cleaned text is not Swiss German, stop and
return an EMPTY `"translations"` list together with the label
`"Not Swiss German"` and confidence level `0`. Do not clean, segment, or
translate it. This applies to the non-Swiss Alemannic varieties listed above
exactly as it does to Standard High German, English, or any other language:
Swabian, Badisch, Alsatian, Vorarlbergisch, and non-Swiss Walser text must
produce no translation segments at all.

Only Swiss German text is translated.


### PART 5 — DIALECT IDENTIFICATION

Classify the complete cleaned text using exactly one of these labels:

* Zürich
* Luzern
* St. Gallen
* Bern
* Basel
* Wallis
* Graubünden
* Swiss German
* Not Swiss German

Return a regional dialect when it can be identified either through your own
linguistic assessment or through the explicit dialect classification rules.

Return `"Swiss German"` when the text is Swiss German, but neither your own
assessment nor the explicit rules can identify a regional dialect reliably.

Return `"Not Swiss German"` when the text is not Swiss German.

---

### PART 6 — DIALECT CONFIDENCE LEVEL

Return one confidence level from 0 to 3.


#### Level 3

Return `3` when:

- Your own linguistic assessment reliably identifies a regional dialect.
- The explicit dialect classification rules independently identify the same regional dialect.
- The complete text contains sufficient and consistent regional evidence.

#### Level 2


Return `2` when:

* Only your own linguistic assessment identifies a regional dialect.

OR:

* Only the explicit dialect classification rules identify a regional dialect.

OR:

* Both methods identify regional dialects, but they identify different dialects.

If the two methods disagree, return the dialect supported by the stronger
evidence across the complete text. Because dialect rules can have exceptions,
give more weight to your own linguistic assessment than to an isolated rule.


#### Level 1

Return `1` when:

* The text is Swiss German.
* No regional dialect can be identified reliably.

The label must be `"Swiss German"`.

#### Level 0

Return `0` when:

* The text is not Swiss German.

The label must be `"Not Swiss German"`.
---

# ROLE AND TASK

You are an expert Swiss German dialect classifier.

Analyze the complete cleaned text and classify it as one of the seven regional
Swiss German dialects, general Swiss German, or Not Swiss German.

Use both:

1. Your general linguistic knowledge.
2. The explicit lexical, phonological, morphological, and orthographic rules
   below.

Treat these as two separate sources of evidence when selecting the
confidence level.

---

## DIALECT CLASSIFICATION RULES

The explicit dialect rules below are indicators, not complete definitions.
A dialect may contain exceptions or use markers that are not listed here.

Apply the rules as follows:

* Use the listed markers to produce the rule-based assessment.
* Use your broader linguistic knowledge to produce your own assessment.
* Do not require every listed marker to be present.
* Do not reject a dialect solely because an expected marker is absent.
* When the two assessments disagree, prefer your broader linguistic assessment,
  but return confidence level `2`.

  
### 1. Basel — Baaseldytsch

#### Vocalism

Basel normally uses "isch", not "esch".

#### Entrundung

The rounded vowels ü and ö may become unrounded i and e.

Indicators:

* grien instead of grün
* scheen instead of schön
* ditsch instead of deutsch
* Minschter instead of Münster

#### Initial K

Initial k may remain an aspirated k or kh instead of becoming a velar ch.

Indicators:

* Khind / Kind instead of Chind
* Khopf / Kopf instead of Chopf

#### Negation and tags

Indicators:

* nit
* sentence-final "nit?"

#### Lexical markers

Possible indicators:

* Gellerettli
* Fazenettli
* Baareblyy
* adee
* doo
* woo

Do not classify a text as Basel solely because it contains "isch" or "nit",
as these forms also occur in other dialects.

---

### 2. Zürich — Zürischnurre

#### Verb plurals

Plural verbs commonly use the ending "-ed".

Indicators:

* mir singed
* ihr singed
* sie singed
* gönd
* händ
* chönd

#### Negation and copula

Common indicators:

* nöd
* isch

"ned" and "esch" are normally not Zürich markers.

#### Monophthongization

Middle High German diphthongs may be represented using long monophthongs.

Indicators:

* Staa
* Saife / Sääfe
* Chlee

#### Diminutives

Common diminutive:

* -li

Example:

* Chindli

---

### 3. Luzern — Lozärnerdütsch / Central Swiss

#### Copula

A strong indicator is:

* esch

#### Negation

Common indicators:

* ned
* neb

#### Vowel lowering

High vowels may be lowered.

Indicators:

* Lozärn
* open-o forms such as Chopf

#### Lexical markers

Possible indicators:

* rüüdig
* öppis
* Statt
* gönd

"esch" is strong evidence for Luzern or Central Swiss and is not a typical
Basel marker.

---

### 4. Bern — Bärndütsch

#### L-vocalization

An l before a consonant or at the end of a syllable may become a vocalic u.

Strong indicators:

* Miuch instead of Milch
* aui instead of alle
* Esu instead of Esel
* Waud instead of Wald
* choufe instead of kaufen
* Outo instead of Auto

L-vocalization is one of the strongest Bernese markers.

#### ND-coalescence

The cluster nd may become ng.

Indicators:

* Ching instead of Chind
* finge instead of finden
* angers instead of anders

#### Articles

Strong indicator:

* ds as a neuter definite article

Examples:

* ds Chind
* ds Outo

#### Verbs and negation

Common indicators:

* isch
* nid
* nit
* infinitive and plural endings in -e

Examples:

* singe
* lache

#### Lexical markers

Possible indicators:

* gäll
* scho
* ebbe
* lue
* lueg

---

### 5. St. Gallen — Eastern Swiss

This category includes eastern Swiss forms, particularly St. Gallen and
closely related Appenzell forms.

#### Infinitive endings

Infinitives and adjective forms may end in a bright "-a".

Indicators:

* macha
* singa
* guata
* schöna

#### Plural verbs

Possible indicators:

* gond
* gand
* hend
* chömed
* händ

#### Vowels

Eastern Swiss writing may contain frequent bright and open ä vowels.

#### Lexical markers

Possible indicators:

* öppä
* gsi
* nöd
* nid

Do not classify a text as St. Gallen from "gsi" alone, as this form is common
throughout much of Switzerland.

---

### 6. Wallis — Wallisertitsch / Highest Alemannic

#### S-to-sch developments

Some words show characteristic sch forms.

Indicators:

* iisch
* Hiischer
* Liisch
* muesch

#### Predicative adjective inflection

Predicative adjectives may retain an ending.

Example:

* "As Hus isch rots."

#### Vowels and vocabulary

Possible indicators:

* highly archaic vowel forms
* zer
* nett
* nit
* Wallise

For the rule-based assessment, identify Wallis only when several distinctive
Highest Alemannic markers are present. Your broader linguistic assessment may
still identify Wallis based on other reliable evidence not listed here.


---

### 7. Graubünden — Bündnerdütsch

#### Archaic diphthongs

Older "uo" forms may be retained where other dialects use "ue".

Indicators:

* guot instead of guet
* Khua instead of Chue

#### Initial K

Initial k may remain aspirated.

Indicator:

* Khind instead of Chind

#### Romansh influence

Possible evidence includes:

* Romansh loanwords
* regional vocabulary
* orthographic forms associated with the Bündner Rhine Valley

#### Verb plurals

Possible indicator:

* chömet

An aspirated initial k alone is insufficient because it may also occur in Basel.

---

## SAFETY GUIDELINES FOR THE CLASSIFIER

1. Do not conflate Basel with Central Switzerland.

   * `"esch"` supports Luzern or Central Swiss.
   * Basel normally uses `"isch"`.

2. Check for Bernese l-vocalization first.

   Strong indicators include:

   * aui
   * Miuch
   * Outo
   * Waud
   * choufe

3. Differentiate common negative particles carefully.

   * nöd → usually Zürich or Eastern Swiss
   * ned → usually Luzern or Central Swiss
   * nit → may indicate Basel, Wallis, or Bern
   * nid → often Bern or Eastern Swiss

   A negative particle alone is not sufficient for regional classification.

4. Evaluate the two dialect-identification methods independently:

   * general linguistic assessment
   * explicit dialect classification rules

5. Do not invent dialect evidence.

6. If the text is Swiss German but neither method can identify a regional
   dialect, return:

   * `"label": "Swiss German"`
   * `"confidence_level": 1`

7. If the cleaned text is not Swiss German, return:

   * `"label": "Not Swiss German"`
   * `"confidence_level": 0`
   * `"translations": []` — an empty list. Never translate non-Swiss German
     text, including non-Swiss Alemannic (Swabian, Badisch, Alsatian,
     Vorarlbergisch, non-Swiss Walser).

8. Never return `"Swiss German"` with confidence level `0`, `2`, or `3`.

9. Never return `"Not Swiss German"` with confidence level `1`, `2`, or `3`.

10. Never return a regional dialect with confidence level `0` or `1`.


---

### OUTPUT FORMAT — STRICT JSON

Return ONLY a valid JSON object with exactly this structure:

{
"translations": [
{
"input": "<cleaned Swiss German segment>",
"output": "<Standard High German translation or 'NA'>"
}
],
"dialect": {
"label": "<Zürich|Luzern|St. Gallen|Bern|Basel|Wallis|Graubünden|Swiss German|Not Swiss German>",
"confidence_level": <integer 0-3>
}
}

Constraints:

* `"label"` must be exactly one of:
  `"Zürich"`, `"Luzern"`, `"St. Gallen"`, `"Bern"`, `"Basel"`,
  `"Wallis"`, `"Graubünden"`, `"Swiss German"`, or
  `"Not Swiss German"`.

* The label and confidence level must follow this exact mapping:

  * Regional dialect + both methods agree → confidence level `3`
  * Regional dialect + exactly one method supports it, or the two methods
    identify different regional dialects → confidence level `2`
  * `"Swiss German"` → confidence level `1`
  * `"Not Swiss German"` → confidence level `0`

* Do not use `"NA"` as a dialect label.

* Do not return dialect probabilities.

* Do not return fine-grained regional dialects.

* Do not include `certainty` or `certainty_highest`.

IMPORTANT: Return strict JSON only.

"""

```

## Licence

Text derives from Wikipedia and is released under **CC BY-SA 4.0**, matching the
source. Translations were produced by a machine-translation model.
