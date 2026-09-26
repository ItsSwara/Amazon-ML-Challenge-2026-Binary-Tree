# Shared error and risk tracker

Use this file for reproducible failures, correctness issues and material run risks.
Keep full terminal logs locally in `logs/` (gitignored). Commit concise summaries,
not raw business records, credentials or huge tracebacks.

## Team workflow

1. Add the next unused `ERR-NNN` ID before investigating. Include the commit,
   exact command/config, dataset scope (synthetic/mock/real), and a short error excerpt.
2. Name the owner and the affected outputs. Retain the issue until resolved; do not
   delete old entries or reuse IDs. Coordinate IDs if two teammates edit simultaneously.
3. Set **In progress** while investigating and **Fixed** when a correction is committed.
4. Set **Verified** only after the original failing case passes; link the test/rerun
   evidence. Track merge/deployment separately: verified on a branch is not deployed on main.
5. An **Open** risk can be an inspected code path without a reproduced full-scale
   failure. Say which it is; do not turn estimates into measured results.

Snapshot below checked against main `a12c79e` on 2026-09-26. Ownership is suggested
by stage, not confirmation that a teammate has accepted the issue.

## Index

| ID | Issue | Status | Suggested owner | Integration state |
|---|---|---|---|---|
| ERR-001 | Header-only scored input crashes | Verified | Sujeet | Fixed in `93fcdff`; not in checked main |
| ERR-002 | Record cache reuses same-count stale data | Verified | Sujeet | Fixed in `93fcdff`; not in checked main |
| ERR-003 | Oversized candidate group grows carry buffer | Verified | Sujeet | Fixed in `93fcdff`; not in checked main |
| ERR-004 | Final assignment/verification memory grows with dataset | Open | Shreyashi / Swara | Inspection finding; full-scale peak unmeasured |
| ERR-005 | Decision evaluation excludes zero-candidate references | Open | Shreyashi / Swara | Inspection finding in checked main |
| ERR-006 | Near-tied scores change rank after TSV parsing | Verified | Sujeet | Fix `f0f0556` is in checked main |
| ERR-007 | Clean install lacks required RapidFuzz dependency | Open | Packaging owner / Swara | Missing from main requirements.txt |

## ERR-001 — Header-only scored input crashes

- **First recorded:** 2026-09-26. **Stage:** inference group reader.
- **Evidence:** reproduced on code based on `a12c79e`; `IndexError: single positional indexer is out-of-bounds` at `refs.iloc[-1]`.
- **Reproduce:** pass a TSV containing only `source1_entity_id`, `candidate_entity_id`, `score` header to `reference_batches(path, 1)`.
- **Impact:** no-candidate runs fail instead of exporting empty lists for all references.
- **Fix:** `93fcdff` skips empty chunks after validating the header.
- **Verification:** header-only reader and complete inference regression tests pass; complete suite 75 passed in the inference review.
- **Integration:** `sujeet/inference-review`; pending merge into checked main.

## ERR-002 — Record cache reuses same-count stale data

- **First recorded:** 2026-09-26. **Stage:** SQLite normalized-record lookup.
- **Evidence:** reproduced; changing a record's name while keeping the row count returned the old name. Switching source paths with equal row counts also returned the old country.
- **Reproduce:** build RecordStore, change normalized Parquet values without changing its row count, then build/reuse the same store path.
- **Impact:** features and country guards can use incorrect records silently.
- **Fix:** `93fcdff` fingerprints source paths, contents and schema; replaces the cache only after a complete rebuild.
- **Verification:** same-count name-change and changed-path/country tests pass.
- **Integration:** `sujeet/inference-review`; pending merge into checked main.

## ERR-003 — Oversized candidate group grows carry buffer

- **First recorded:** 2026-09-26. **Stage:** streaming candidate input.
- **Evidence:** a 51-row reference group was accepted despite the agreed Top-50 contract; larger groups could accumulate across reads.
- **Reproduce:** one S1 with 51 pairs, read at chunk sizes 1 or 10.
- **Impact:** malformed inputs can defeat the intended per-group memory bound.
- **Fix:** `93fcdff` checks carried and completed group sizes against the default limit 50.
- **Verification:** both chunk-size regressions reject the input. Normal group-boundary tests still pass.
- **Integration:** `sujeet/inference-review`; pending merge into checked main. Coordinate any intentional Top-K contract change.

## ERR-004 — Final inference memory is not bounded by batch size

- **First recorded:** 2026-09-26. **Stage:** final assignment/output verification.
- **Evidence type:** static inspection, not a reproduced production OOM.
- **Location:** `src/entity_resolution/inference/run_inference.py`: reload of `accepted_pairs.parquet`; `read_id_lists`/`verify_outputs` loading candidate lists into Python sets.
- **Impact:** batch scoring can succeed but full output finalization can exceed RAM. Test upper bound at K=50 is 86,627,200 pairs.
- **Next action:** measure representative acceptance/memory or use disk-backed assignment/verification. Record machine RAM, input count, accepted count and peak memory.
- **Verification needed:** completed representative/full run with measured memory and valid outputs.
- **Caveat:** the successful 2-million-row synthetic test shared only 50 targets; its ~386 MiB peak is not a production estimate.

## ERR-005 — Decision report drops zero-candidate references

- **First recorded:** 2026-09-26. **Stage:** decision evaluation.
- **Evidence type:** static inspection.
- **Location:** `src/entity_resolution/inference/validate_decision.py` restricts truth to `pairs.source1_entity_id.unique()`.
- **Impact:** references absent from scored pairs leave the F0.5 denominator; the resulting score may differ from the complete intended holdout.
- **Next action:** obtain all intended holdout references from frozen split/country metadata and retain them with empty predictions.
- **Verification needed:** a test with a zero-candidate nonsingleton and singleton; macro F0.5 must include both.

## ERR-006 — Near-tied score precision changes ranks on replay

- **First recorded:** 2026-09-26. **Stage:** scored-feature parsing.
- **Evidence:** direct float scores `0.30000000000000004` and `0.3` produced different ranks after string conversion plus pandas `to_numeric`.
- **Impact:** candidate rank/gap can vary between direct execution and TSV replay.
- **Fix:** `f0f0556`, round-trip-preserving float64 conversion in `features/scored.py`.
- **Verification:** decimal-rank regression test; all 100,000 mock feature rows compared across original run and corrected TSV replay.
- **Integration:** included in checked main `a12c79e`.

## ERR-007 — Main install instructions omit RapidFuzz

- **First recorded:** 2026-09-26. **Stage:** environment setup/packaging.
- **Evidence type:** dependency inspection; `features/baseline.py` imports `rapidfuzz`, but main `requirements.txt` omits it. `requirements-features.txt` includes it but omits FAISS.
- **Impact:** README's clean-install path is incomplete for the whole pipeline.
- **Next action:** define a complete, tested dependency installation path; pin the validated CPU environment and document the separate GPU FAISS environment if used.
- **Verification needed:** clean environment install followed by imports, full tests and the intended entry-point smoke run. Record Python and dependency versions.

## New issue template

Copy below, assign the next ID and add an index row:

```markdown
## ERR-NNN — Short specific title

- **Recorded / last updated:** YYYY-MM-DD; timezone.
- **Status:** Open / In progress / Fixed / Verified.
- **Owner / stage:** Name; pipeline stage.
- **Evidence type:** Reproduced failure / code inspection / unverified report.
- **Run:** commit, command, config, environment, synthetic/mock/real scope and run ID.
- **Error:** short exact excerpt or observed wrong behavior.
- **Impact:** affected files/results; whether a completed run must be regenerated.
- **Reproduce:** minimum steps/input shape; no raw business records in the tracker.
- **Local log:** logs/<run-id>.log (share separately when needed).
- **Fix:** explanation + commit/PR, or next investigation step.
- **Verification:** original failing case, command and observed result.
- **Integration:** branch/main state, verified commit and any required rerun.
```

## Save a PowerShell log

From the repository root, start recording before running the relevant pipeline command:

```powershell
New-Item -ItemType Directory -Force -Path logs | Out-Null
$runLog = "logs/run-$(Get-Date -Format 'yyyyMMdd-HHmmss').log"
Start-Transcript -Path $runLog
# Run the actual command here, then record its exit status immediately:
# $LASTEXITCODE
Stop-Transcript
```

Only transcript relevant commands; do not record login/secrets. Add the real command
and result to the issue entry. `logs/` contents remain local; the shared tracker does
not contain the log itself. Push tracker updates on your own branch for review.
