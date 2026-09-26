# Inference feature boundary review — Sujeet, 26 September 2026

Reviewed main `a12c79e`; work is on `sujeet/inference-review`. This review covers
the feature stage's use inside `inference/run_inference.py` and the feature-methodology
section. It does not certify full-scale inference or model accuracy.

## Verified fixes

| Issue reproduced before fix | Correction | Regression evidence |
|---|---|---|
| Header-only scored TSV raised IndexError at `refs.iloc[-1]` | Validate header, then skip empty chunks | Reader test plus complete inference test produces every S1 with empty candidate/match lists |
| SQLite record cache reused old text/country when new normalized inputs had the same number of rows | Cache manifest includes resolved input paths, SHA-256 contents and record schema; legacy caches rebuild | Same-count changed-name and changed-path/country tests now pass |
| A malformed reference group could grow the carry buffer without a limit | Enforce the agreed Top-50 contract on carried and completed groups | 51-row group fails across chunk sizes 1 and 10 |

Cache builds now keep the old database until its replacement is complete and close
the build connection on failure. Source fingerprints are checked before/after building;
inputs changing mid-build fail loudly. Reuse requires one full hash read of the normalized
files; rebuilding hashes before and after ingestion. Freeze those inputs during inference.
This deliberate I/O cost prevents stale features without relying on row counts alone.

## Feature integration checks

- Complete S1 groups are preserved across reads; ranks/gaps are calculated on the whole group.
- SQLite lookup, ID joins, float32 feature order, all 15 feature values and calibrated
  probabilities match direct complete-group calculation, including a missing-address reference.
- Header-only candidates still yield complete official-format output lists.
- Wrong headers, invalid groups, ID errors and NaN behavior are covered by tests.
- Entire repository suite: **75 passed**. Test environment: Python 3.12.14,
  pandas 2.2.3, sklearn 1.6.1, LightGBM 4.7.0, FAISS CPU 1.15.1.

## Synthetic streaming stress run

Reproduction, from repository root with dependencies installed:

```powershell
python scripts/stress_inference_features.py --output artifacts/streaming_review_2m
```

Use a fresh output directory. This script creates synthetic data only. It calls the
actual SQLite store, group reader, feature extractor, calibrated LightGBM scoring,
decision and output verification code. Chunk size 50,003 intentionally cuts across
50-pair reference groups. It also includes missing addresses.

| Measurement | Observed |
|---|---:|
| Candidate rows scored and preserved in export | 2,000,000 |
| References preserved | 40,000 |
| Distinct targets | 50 |
| Scoring time | 67.6 seconds |
| Complete stress script time | 71.376 seconds |
| Scoring throughput | 29,594.5 pairs/second |
| Peak Python process working set | 405,053,440 bytes (~386 MiB) |
| Pairs passing threshold before global assignment | 400,000 |

The toy classifier's matches are not accuracy evidence. Only 50 targets are shared
across many references: this tests multi-million-row streaming and repeated batch
execution, not storage for ten million distinct target records, full string diversity,
GPU/blocker memory or final-run elapsed time. Do not extrapolate the 386 MiB result
to production. Detailed local evidence is `artifacts/streaming_review_2m/stress_summary.json`.

## Remaining integration risks for team owners

1. **Full inference is not wholly memory-bounded.** Feature scoring uses batches,
   but finalization reloads every threshold-accepted pair into a DataFrame, and
   `verify_outputs` loads all candidate lists into Python sets. The real test upper
   bound is 86,627,200 candidate pairs. The closed-reference/seen/S1 sets also grow
   with reference count. Measure memory or move finalization/verification to disk-backed
   processing before describing the full path as bounded. The code's opening comment
   now states this limit explicitly; these finalization algorithms were not redesigned
   in a feature-boundary patch.
2. **LOCO preprocessing must match the claim.** The current handoff fits word IDF on
   the random training subset (both countries); `models/loco.py` then pools mock splits
   and trains the booster/calibrator on US. This is not fully US-only preprocessing.
   Refit/recompute on the US training fold for a strict country-held-out experiment,
   or describe the transductive preprocessing limitation accurately.
3. **Complete evaluation denominator.** `inference/validate_decision.py` selects truth
   using references present in scored pairs, excluding zero-candidate references if
   any occur. The decision report should use the complete intended country holdout.
4. **Artifact identity/provenance.** Inference defaults to `artifacts/matcher_loco`
   and the mock handoff extractor path. The matcher bundle checks column names but
   does not bind the extractor hash or record its exact fit corpus. Before a full run,
   confirm and explicitly pass the mutually compatible model, extractor and threshold;
   a valid 15-column shape does not establish the correct fitted vocabulary.
5. **Reported numbers need run artifacts.** Threshold 0.68 and F0.5 0.9625 were supplied
   in the team message; this review did not reproduce or independently verify that run.
   They were not copied into the methodology as final/full-data results.

## Corrections to the real-data handoff message

Based on the committed full audit:

- Test S1 output must contain **1,732,544 data rows** in each official list-format
  file (**1,732,545 lines including the header**). The scored long file has one row
  per pair, not one row per reference; at K=50 its upper bound is **86,627,200 pairs**.
- There is no test ground truth. Report real-train/frozen-validation blocking recall;
  test gets coverage/count/runtime checks, not a measurable test recall claim.
- Train is not much smaller: train has 2,206,821 references and 10,320,219 pooled
  targets; test has 1,732,544 references and 9,969,589 targets. Budget both runs.
- The stated 42 GB peak is an estimate from the team message, not a measurement
  reproduced here. Keep measured peak memory and configuration in the run report.
- CLI module commands need the repository `src` on Python's import path; the direct
  script entry points already set it where provided.

## Methodology deliverable

`Documentation_template.md` was copied from the provided student_resource directory.
Section 4 now contains the complete 15-feature list and one-line rationale, exact
definitions, missingness policy, fitting/reuse policy and streaming contract.
The rest remains explicitly a draft for the responsible teammates to finalize.
Advanced address features remain dependent on the address parser and error analysis.
