# ML Challenge 2026: Business Entity Resolution Solution Template

> Working draft: the feature-engineering subsection is completed by Sujeet.
> Other sections and final model/results must be completed and verified by their
> respective owners before submission. Placeholder fields are not final claims.

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
*Provide a brief 2-3 sentence overview of your approach and key innovations.*

---

## 2. Methodology

### 2.1 Problem Analysis
*Key insights discovered during EDA — noise patterns, address variations, missing fields, etc.*

### 2.2 Solution Strategy
*Outline your high-level approach.*

**Approach Type:** [Blocking + Classifier / End-to-End / Graph-Based / Hybrid, etc]  
**Core Innovation:** [Brief description of your main technical contribution]

---

## 3. Candidate Generation (Blocking)
*Describe how you reduced the comparison space to a manageable candidate set.*

- **Blocking keys used:** [e.g., PIN code, phonetic name encoding, TF-IDF, etc.]
- **Candidate pairs generated:** [total]
- **How you ensured true matches were not lost:**

---

## 4. Matching Model

**Features used:**

The baseline uses the ordered 15-column `scored-baseline-v2` contract below. Each
row corresponds to a candidate pair surviving blocking. `source1_entity_id` and
`candidate_entity_id` are join keys, never model inputs. Labels are stored separately.

| Feature | One-line rationale |
|---|---|
| `name_jaro_winkler` | Character agreement and common-prefix evidence for spelling variants. |
| `name_token_sort` | Detects matching name words despite changes in word order. |
| `name_tfidf_cosine` | Weights name-token agreement by rarity in the fitted training corpus. |
| `name_idf_overlap` | Measures weighted shared name tokens relative to their union. |
| `address_jaro_winkler` | Captures character-level agreement between normalized addresses. |
| `address_token_sort` | Tolerates reordering of address words and numbers. |
| `address_tfidf_cosine` | Gives uncommon shared address tokens more influence than common tokens. |
| `address_idf_overlap` | Measures IDF-weighted address-token intersection over union. |
| `source1_name_missing` | Distinguishes an absent reference name from a dissimilar name. |
| `source1_address_missing` | Identifies missing reference address evidence. |
| `candidate_name_missing` | Identifies missing candidate name evidence. |
| `candidate_address_missing` | Identifies missing candidate address evidence. |
| `blocking_score` | Carries the blocker's combined name/address similarity signal into the matcher. |
| `candidate_rank` | Expresses relative retrieval position among this reference's S2/S3 candidates. |
| `blocking_top1_top2_gap` | Indicates how separated the two strongest retrieval scores are. |

**Definitions and ranges.** Jaro-Winkler is normalized to [0, 1]; RapidFuzz
token-sort ratio is divided by 100. Name and address use separate word-unigram
TF-IDF vectorizers with whitespace tokenization, L2 normalization, smoothed IDF
and a maximum of 100,000 vocabulary entries per field. Weighted overlap is the
sum of fitted IDFs over the shared token set divided by the sum over the union.
Unknown/pruned tokens have weight zero; a nonmissing zero-denominator comparison
returns 0. All model features are exported as float32.

**Retrieval features.** Higher blocking score is better:
`max(name_char_cosine, address_char_cosine) + 0.05` when at least one token is
shared. Its range is [0, 1.05], not a probability. Rank is one-based, ordered by
descending score then ascending candidate ID for ties, across the combined S2/S3
group. The top-two gap is the highest minus second-highest score and is repeated
for every candidate in that reference group. It is NaN when there is only one
candidate. Ranks/gaps are computed on complete reference groups before batching.

**Missing values and normalization.** Inputs are the data team's normalized
`name_norm`/`address_norm`. Raw placeholder rules are not applied a second time:
for example, a genuine business name may normalize to the literal string `nan`.
If either side of a field is missing, all four similarities for that field are
NaN, accompanied by the explicit 0/1 missingness flags. Two missing fields never
produce a perfect similarity. LightGBM consumes NaNs directly. Legal suffix and
country are not part of this 15-column feature matrix; country is used downstream
by decision logic.

**Fitting and reuse.** In the completed mock handoff, word TF-IDF/IDF is fitted
only on the frozen training references and their labeled matched targets, excluding
held-out reference-owned targets and unassigned distractors. The same saved
extractor transforms validation and inference inputs; transformation never refits.
This policy is distinct from the blocker's target-pool character IDF. A strict
country-held-out experiment must fit feature preprocessing on its country-training
fold too; the random-split mock extractor must not be described as US-only.
The final production fit corpus/artifact must be recorded with the model owner.

**Streaming interface.** Inference reads grouped scored pairs, carries incomplete
reference groups between chunks, computes rank/gap before feature extraction,
looks up only batch IDs in SQLite, and passes the ordered 15 columns to the model.
The reviewed default contract caps groups at 50 candidates. Unit tests exercise
nulls, unknown/duplicate IDs, score precision, group boundaries, saved extractor
reuse and cache invalidation. This describes the feature-scoring path; full-run
memory also depends on final assignment and output verification.

**Scope.** No external lookup, entity database, API or geocoding is used by these
features. Component-level address features and other advanced signals are deferred
until the data team's address parser and model error evidence are available.

**Model type:** [e.g., XGBoost, Siamese Network, Transformer, etc.]  
**Threshold selection method:** [e.g., F_0.5 optimization on validation set]

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [your best validation score]
- **Common false positives (wrong merges):** [brief description]
- **Common false negatives (missed matches):** [brief description]

---

## 6. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

---

## Appendix

### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
