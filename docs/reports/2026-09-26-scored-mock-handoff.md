# Actual-blocking mock handoff verification — 26 September 2026

Yash's blocker (`fd8fbcc`, merged into the feature branch) was executed unchanged
through the new integration runner on the repository's linked normalized training
mock. Feature schema: `scored-baseline-v2`, 15 float32 features. Top-K: 50 across S2/S3.

| Measure | Training | Validation |
|---|---:|---:|
| Frozen-split S1 references | 1,586 | 414 |
| Retrieved candidate pairs / feature rows | 79,300 | 20,700 |
| True links in complete ground truth | 5,531 | 1,457 |
| Retrieved positive links | 5,502 | 1,453 |
| Missed true links | 29 | 4 |
| Negative candidate pairs | 73,798 | 19,247 |
| Micro blocking recall | 99.4757% | 99.7255% |
| Macro recall over nonsingletons | 99.4622% | 99.7449% |
| True singleton references | 69 | 22 |
| Pairs with missing-address similarities | 2,266 | 560 |

All 2,000 references received 50 candidates; zero were dropped. Missing-address
similarities remain NaN with a matching missingness flag. Labels are separate.
Complete ground truth retains links missed by blocking for downstream F0.5 evaluation.

## Verification

- 26 automated tests passed, covering text features, nulls, ID validation, training
  corpus exclusion, tie handling, single-candidate gap, empty exports, scorer denominators,
  label joins, frozen reference coverage, failed-run guards and decimal precision.
- All 100,000 rows were independently checked against truth labels and candidate-list exports.
- Batched output at 1,000 pairs matched a TSV replay at 137 pairs, including rank/gap.
- Reloaded extractor reproduced sampled text features for both splits.
- Every generated numeric feature is float32; no infinities; missingness checks agree.
- Run manifests record input/output SHA-256 hashes, code hashes and package versions.

A replay check initially exposed that pandas `to_numeric` can round near-tied decimal
scores differently from the original float64 values, changing ranks. The scored input
adapter now uses float64 conversion that preserves the decimal round trip; a regression
test reproduces the previous rank change. The final handoff uses the corrected path.

## Timing and scope

Original retrieval + feature export: **168.148 seconds** locally. Reusing its scored
TSV with the final parser and batch size 137: **19.921 seconds**. These are single
machine observations, not a production runtime estimate. Final local artifact folder:
`artifacts/scored_mock_v2_final/`. Older replay folders are not the handoff.

Blocking IDF is fitted on the full mock target pool without labels, including targets
owned by validation references. Feature IDF is fitted only on frozen training references
and their true matched targets (7,117 records). These policies are distinct and recorded.

This is actual retrieval on a small, selected, linked mock with a much smaller target
pool than production. **It does not establish full-data recall, final F0.5 or France
generalization.** No matcher was trained. Full-scale retrieval and on-disk record lookup
are still pending. See [handoff instructions](../schemas/scored_features_v2.md).
