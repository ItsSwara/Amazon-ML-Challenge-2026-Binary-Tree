# Normalized column schema

Contract for the output of `src/entity_resolution/normalization/` (built in step 2). Build against
this now; the mock sample in `data/mock/` has the raw schema, and the normalizer can be run on it
to produce these columns.

## Mock data (`data/mock/`)

A small linked sample of the 7 dataset files, rows copied byte-for-byte with the original filenames
(`train_source1/2/3.tsv`, `train_ground_truth.tsv`, `test_source1/2/3.tsv`), flat in one folder.
Row order within each file is shuffled (fixed seed). Regenerate with
`python src/entity_resolution/data/make_mock.py` (`--refs`, `--seed` to change).

### Train side: fully linked

| File | Rows | Contents |
|---|---|---|
| `train_source1.tsv` | 2,000 | The 2,000 lowest-numbered references (by numeric ID). 1,202 US / 798 India. |
| `train_ground_truth.tsv` | 2,000 | Their ground-truth rows, verbatim. 91 references (4.55%) have no matches; 5.55% have 1; 89.9% have 2+. |
| `train_source2.tsv` | 6,706 | 3,353 rows matched to those references + 3,353 random rows matched to none of them. |
| `train_source3.tsv` | 7,270 | 3,635 rows matched to those references + 3,635 random rows matched to none of them. |

- **Match rate: 100% of the 6,988 matched IDs in the mock ground truth are present in the mock
  S2/S3 files** (not ~0.2%, as in the earlier head-of-file mock). You can compute real precision,
  recall and F0.5 on it.
- About half of each mock S2/S3 file is matched; the other half are genuine negatives (rows that
  belong to none of the 2,000 references). The real files are about 73% matched, so a mock-trained
  model sees a higher negative rate than production.
- The negatives are any S2/S3 rows not matched to the 2,000 mock references. About 73% of them
  belong to *other* references in the full dataset, so they are realistic distractors, not junk.
- The 2,000 references are the lowest-numbered IDs, not a uniform random sample. Country and match
  count mix are close to the full data (89.9% have 2+ matches vs 89.0%), but treat any statistic
  from 2,000 references as noisy.

### Test side: NOT reliably linked

There is no test ground truth, so the test mock cannot be linked the way the train mock is.
`test_source1.tsv` holds the 2,000 lowest-numbered test references (931 India, 757 US, 312 France).
`test_source2.tsv` (5,642 rows) and
`test_source3.tsv` (5,867 rows) are uniform random samples sized in proportion to the full files
(2,000 / 1,732,544 of each). Because they are random, almost none of the true matches for the 2,000
test references are in them. Use the test mock to check parsing, normalization and inference plumbing
(including France text), never to estimate match quality.

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
2. Repair mojibake (UTF-8 text mis-decoded as Latin-1, e.g. `Ã‚\x80\x93` for an en-dash,
   `Ã¢\x80\x99` for an apostrophe). Tries latin-1 encode then utf-8 decode on the whole string;
   if that raises (mixed strings, or title-casing turned lead byte `Ã¢` into `Ã‚`), repairs just the
   mojibake runs. Undecodable text is left unchanged, and clean text (`PrÃ©sident`, Devanagari) is
   never altered. Must run before step 3 because the continuation bytes are C1 control chars.
3. Replace control characters `[\x00-\x1F\x7F-\x9F]` with a space.
4. `unidecode` transliteration to ASCII (Devanagari, accented French, etc.).
5. Lowercase.
6. `&` becomes ` and `. Apostrophes are deleted (`mcdonald's` to `mcdonalds`). Dots between
   single letters are deleted (`l.l.c.` to `llc`).
7. Every other non-alphanumeric character becomes a space; whitespace is collapsed and trimmed.
8. Names only: trailing legal-form tokens (see `LEGAL_SUFFIXES` in
   `normalization/normalize.py`) are moved to `legal_suffix` in canonical form (`corporation` to
   `corp`, `limited` to `ltd`, `private limited` to `pvt ltd`, ...). A name that would be left empty
   keeps its text and gets NaN suffix.
9. Empty result becomes NaN.

## Ground truth in code

Represent ground truth as `dict[str, set[str]]`: `source1_entity_id` to the set of matched IDs.
A reference with no matches maps to an empty set (`set()`), not `None`. Predictions use the same
shape. See `evaluation/scorer.py`.
