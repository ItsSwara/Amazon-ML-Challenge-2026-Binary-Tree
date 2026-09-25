"""Macro F_0.5 scorer for entity-resolution predictions.

    score(predictions, ground_truth) -> float

Both arguments map a source1 reference ID to the set of matched S2/S3 IDs.
Every reference in ground_truth is scored; a reference missing from
predictions counts as an empty prediction, and predictions for references not
in ground_truth are ignored.

Run the built-in checks:  python src/entity_resolution/evaluation/scorer.py
"""

import math

BETA = 0.5


def reference_score(predicted: set[str], truth: set[str], beta: float = BETA) -> float:
    """F_beta for one reference. No matches expected: 1.0 if nothing predicted, else 0.0."""
    if not truth:
        return 1.0 if not predicted else 0.0
    tp = len(predicted & truth)
    if tp == 0:  # also covers empty prediction; avoids 0/0
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(truth)
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def score(predictions: dict[str, set[str]], ground_truth: dict[str, set[str]]) -> float:
    """Macro-average F_0.5 over all references in ground_truth."""
    if not ground_truth:
        raise ValueError("ground_truth is empty")
    total = sum(
        reference_score(set(predictions.get(ref, ())), set(truth))
        for ref, truth in ground_truth.items()
    )
    return total / len(ground_truth)


def _run_checks() -> bool:
    # (description, predicted, truth, expected score)
    cases = [
        ("singleton predicted correctly (truth empty, pred empty)", set(), set(), 1.0),
        ("singleton predicted incorrectly (truth empty, pred has 1)", {"S2-1"}, set(), 0.0),
        # tp=2, P=2/3, R=2/4=1/2 -> 1.25 * (1/3) / (0.25*(2/3) + 1/2) = 0.625
        ("multi-match, partial overlap (2 of 4 found, 1 wrong)",
         {"S2-1", "S3-2", "S2-9"}, {"S2-1", "S3-2", "S2-3", "S3-4"}, 0.625),
        ("multi-match, perfect", {"S2-1", "S3-2"}, {"S2-1", "S3-2"}, 1.0),
        ("multi-match, nothing predicted", set(), {"S2-1", "S3-2"}, 0.0),
        ("multi-match, disjoint prediction", {"S2-8"}, {"S2-1"}, 0.0),
        # P=1, R=1/2 -> 1.25 * 0.5 / (0.25 + 0.5) = 5/6: precision counts more than recall
        ("multi-match, precise but incomplete (1 of 2 found)", {"S2-1"}, {"S2-1", "S3-2"}, 5 / 6),
        # P=1/2, R=1 -> 1.25 * 0.5 / (0.125 + 1) = 5/9
        ("multi-match, complete but imprecise (2 found + 2 wrong)",
         {"S2-1", "S3-2", "S2-8", "S2-9"}, {"S2-1", "S3-2"}, 5 / 9),
    ]
    ok = True
    print(f"{'case':<60}{'expected':>10}{'actual':>10}")
    for desc, pred, truth, expected in cases:
        actual = reference_score(pred, truth)
        passed = math.isclose(actual, expected, abs_tol=1e-9)
        ok &= passed
        print(f"{desc:<60}{expected:>10.4f}{actual:>10.4f}  {'PASS' if passed else 'FAIL'}")

    # Macro over the first four cases: (1 + 0 + 0.625 + 1) / 4, and it differs from pooled.
    truth = {"r1": set(), "r2": set(), "r3": cases[2][2], "r4": cases[3][2]}
    pred = {"r1": set(), "r2": {"S2-1"}, "r3": cases[2][1], "r4": cases[3][1]}
    expected = (1 + 0 + 0.625 + 1) / 4
    actual = score(pred, truth)
    passed = math.isclose(actual, expected, abs_tol=1e-9)
    ok &= passed
    print(f"{'macro over 4 references (r1..r4)':<60}{expected:>10.4f}{actual:>10.4f}  {'PASS' if passed else 'FAIL'}")

    # A reference missing from predictions counts as empty; extra predictions are ignored.
    actual = score({"extra": {"S2-1"}}, {"r1": set(), "r2": {"S2-1"}})
    passed = math.isclose(actual, 0.5, abs_tol=1e-9)
    ok &= passed
    print(f"{'missing prediction = empty; extras ignored':<60}{0.5:>10.4f}{actual:>10.4f}  {'PASS' if passed else 'FAIL'}")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _run_checks() else 1)
