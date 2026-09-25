# Normalized column schema

Contract for the output of `src/entity_resolution/normalization/` (built in step 2). Build against
this now; the mock sample in `data/mock/` has the raw schema, and the normalizer can be run on it
to produce these columns.

## Mock data (`data/mock/`)

The first 10,000 data rows of each of the 7 dataset files, byte-for-byte, same filenames
(`train_source1/2/3.tsv`, `train_ground_truth.tsv`, `test_source1/2/3.tsv`), flat in one folder.
Regenerate with `python src/entity_resolution/data/make_mock.py`.

> Caveat: the dataset files are not sorted by ID or linked by position. In the mock, the ground
> truth rows are a different 10k references than the `train_source1` rows, and only ~0.2% of the
> matched S2/S3 IDs appear in the mock S2/S3 files. Use the mock to exercise parsing, normalization
> and schemas, not to measure match quality.

## Raw schema (unchanged)

Tab-separated, UTF-8, header row, no quoting (`"` is an ordinary character).

| File | Columns |
|---|---|
| `*_source1/2/3.tsv` | `entity_id`, `business_name`, `business_address`, `country` |
| `train_ground_truth.tsv` | `source1_entity_id`, `matched_entity_ids` (comma-separated `S2-`/`S3-` IDs; empty = no matches) |

Read raw files with `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)`
so pandas does not do its own NaN coercion or quote handling; the normalizer owns that.

## Normalized source tables

One row per input row, same row count and order as the input. Identical for every source file
(S1, S2, S3; train and test). Stored as Parquet under `data/processed/normalized/<file>.parquet`
(`normalized_mock/` for the mock), all columns pandas string dtype with NaN as the missing value.
Regenerate with `python src/entity_resolution/normalization/normalize.py --source mock|real`.

| Column | Type | Missing? | Description |
|---|---|---|---|
| `entity_id` | string | never | Verbatim from input, e.g. `S2-166376419`. |
| `name_norm` | string | NaN if name missing | Business name after normalization, **legal suffix removed**. |
| `legal_suffix` | string | NaN if none found | Canonical legal-form tokens stripped from the end of the name, space-joined in original order, e.g. `pvt ltd`, `llc`, `sarl`. |
| `address_norm` | string | NaN if address missing | Business address after normalization. |
| `country` | string | never | Verbatim label (`US`, `India`, `France`); not lowercased, used as a categorical key. |

### NaN convention

- Missing is a true NaN (`pd.NA`/`np.nan`), **never** the strings `""`, `"null"`, `"nan"`, `"-"`
  and **never** `0` or any other imputed value.
- Input values that are empty/whitespace, or one of `null`, `nan`, `none`, `n/a`, `na`, `nil`, `-`,
  `--`, `\N` (any case, surrounding whitespace ignored), become NaN. This is the same set the
  Stage 1 audit counted as null-like. Only whole-field matches count: a business genuinely named
  "Nan Inc." stays `name_norm="nan"`, `legal_suffix="inc"`.
- A field that is empty after normalization (e.g. a name that was only punctuation) becomes NaN.
- A missing name gives NaN in both `name_norm` and `legal_suffix`. `legal_suffix` is also NaN when
  the name has no recognised suffix; it is not an error and not the empty string.
- Consumers must handle NaN explicitly (`.isna()`), not assume strings.

### Normalization steps (applied in this order, to `business_name` and `business_address`)

1. Null-token check on the raw value (see above).
2. Replace control characters `[\x00-\x1F\x7F-\x9F]` with a space.
3. `unidecode` transliteration to ASCII (Devanagari, accented French, etc.).
4. Lowercase.
5. `&` becomes ` and `. Apostrophes are deleted (`mcdonald's` to `mcdonalds`). Dots between
   single letters are deleted (`l.l.c.` to `llc`).
6. Every other non-alphanumeric character becomes a space; whitespace is collapsed and trimmed.
7. Names only: trailing legal-form tokens (see `LEGAL_SUFFIXES` in
   `normalization/normalize.py`) are moved to `legal_suffix` in canonical form (`corporation` to
   `corp`, `limited` to `ltd`, `private limited` to `pvt ltd`, ...). A name that would be left empty
   keeps its text and gets NaN suffix.
8. Empty result becomes NaN.

## Ground truth in code

Represent ground truth as `dict[str, set[str]]`: `source1_entity_id` to the set of matched IDs.
A reference with no matches maps to an empty set (`set()`), not `None`. Predictions use the same
shape. See `evaluation/scorer.py`.
