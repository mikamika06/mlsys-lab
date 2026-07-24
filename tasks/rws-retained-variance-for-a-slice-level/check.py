import numpy as np

def _oracle(eigenvalues, s):
    """Recompute (k, retained_ratio) from scratch using NumPy."""
    ev = np.asarray(eigenvalues, dtype=np.float64)
    ev = np.sort(ev)[::-1]                     # descending
    total = float(ev.sum())
    if total <= 0.0:
        return (0, 0.0)
    cumsum = np.cumsum(ev)
    fractions = cumsum / total
    # smallest k (1-indexed) with fraction >= s
    idx = int(np.searchsorted(fractions, s, side="left"))
    k = min(idx + 1, len(ev))
    retained = float(cumsum[k - 1] / total)
    return (k, retained)

def grade(sol, fx) -> dict:
    test_cases = [
        ([10.0, 5.0, 3.0, 1.0, 0.5], 0.5),
        ([10.0, 5.0, 3.0, 1.0, 0.5], 0.9),
        ([10.0, 5.0, 3.0, 1.0, 0.5], 1.0),
        ([10.0, 5.0, 3.0, 1.0, 0.5], 0.1),
        ([5.0, 4.0, 3.0, 2.0, 1.0], 0.6),
        ([1.0], 1.0),
        ([1.0, 1.0, 1.0, 1.0, 1.0], 0.5),
        ([100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.5),
        (np.linspace(10, 1, 20).tolist(), 0.75),
        (np.logspace(2, 0, 50).tolist(), 0.95),
        (np.logspace(3, 0, 100).tolist(), 0.99),
        ([0.0, 0.0, 0.0], 0.5),
        ([20.0, 0.0, 0.0, 0.0], 0.5),
    ]

    # Append any fixture-based cases
    if fx and "fixtures" in fx:
        for entry in fx["fixtures"]:
            test_cases.append((entry["eigenvalues"], entry["s"]))

    k_ok_count = 0
    max_ratio_err = 0.0

    for ev, s in test_cases:
        try:
            k_learner, ratio_learner = sol.retained_variance_for_slice(ev, s)
            k_learner = int(k_learner)
            ratio_learner = float(ratio_learner)
        except Exception:
            return {"k_accuracy": 0.0, "ratio_pass": 0.0}

        k_oracle, ratio_oracle = _oracle(ev, s)

        if k_learner == k_oracle:
            k_ok_count += 1

        if ratio_oracle == 0.0:
            r_err = 0.0 if ratio_learner == 0.0 else 1.0
        else:
            r_err = abs(ratio_learner - ratio_oracle) / abs(ratio_oracle)
        max_ratio_err = max(max_ratio_err, r_err)

    n = len(test_cases)
    k_accuracy = 1.0 if k_ok_count == n else k_ok_count / n
    ratio_pass = 1.0 if max_ratio_err < 1e-6 else 0.0

    return {"k_accuracy": k_accuracy, "ratio_pass": ratio_pass}
