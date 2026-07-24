import numpy as np

from mlsys import scorers


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _oracle_expected_acceptance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute the expected single-token acceptance probability two
    independent, mathematically equivalent ways, and cross-check them
    against each other before trusting the result as the reference.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    via_overlap = float(np.sum(np.minimum(p, q)))
    via_total_variation = float(1.0 - 0.5 * np.sum(np.abs(p - q)))

    assert abs(via_overlap - via_total_variation) < 1e-9, "identity broken on fixture"
    return via_overlap


def _extra_cases():
    rng = np.random.default_rng(3)
    cases = []
    for n in (2, 5, 20):
        logits_p = rng.normal(0.0, 2.0, size=n)
        logits_q = logits_p + rng.normal(0.0, 1.5, size=n)
        cases.append((_softmax(logits_p), _softmax(logits_q)))
    # identical distributions -> acceptance must be exactly 1.0
    same = _softmax(rng.normal(size=6))
    cases.append((same, same.copy()))
    # disjoint support -> acceptance must be exactly 0.0
    p_disjoint = np.array([0.5, 0.5, 0.0, 0.0])
    q_disjoint = np.array([0.0, 0.0, 0.5, 0.5])
    cases.append((p_disjoint, q_disjoint))
    return cases


def grade(sol, fx) -> dict:
    cases = [(np.asarray(fx["p"]), np.asarray(fx["q"]))] + _extra_cases()

    worst = 0.0
    for p, q in cases:
        ref = _oracle_expected_acceptance(p, q)
        try:
            got = sol.expected_acceptance(p.copy(), q.copy())
            got = float(got)
        except Exception:
            return {"rel_err": float("inf")}

        if not np.isfinite(got):
            return {"rel_err": float("inf")}

        err = scorers.rel_err(np.array([ref]), np.array([got]))
        worst = max(worst, err)

    return {"rel_err": worst}
