import numpy as np


def _cost(L):
    """Causal prefill cost model: position i costs FLOPs proportional to
    i (it attends over i prior tokens), so a request of length L costs
    sum_{i=1}^{L} i = L(L+1)/2."""
    L = np.asarray(L, dtype=np.float64)
    return L * (L + 1) / 2.0


def _oracle(lengths, reused_prefix):
    lengths = np.asarray(lengths, dtype=np.float64)
    reused_prefix = np.asarray(reused_prefix, dtype=np.float64)

    full_cost = _cost(lengths)
    # With `reused_prefix` tokens already cached, only positions
    # reused_prefix+1 .. L are actually prefilled; each of those still
    # costs its own index i in FLOPs (it attends across the whole
    # context including the reused, cached keys).
    reuse_cost = full_cost - _cost(reused_prefix)

    total_full = float(np.sum(full_cost))
    total_reuse = float(np.sum(reuse_cost))
    return 1.0 - total_reuse / total_full


def _cases():
    rng = np.random.default_rng(0)
    cases = []

    for n in [1, 4, 10]:
        lengths = rng.integers(8, 512, size=n)
        reused_prefix = np.array(
            [rng.integers(0, L + 1) for L in lengths], dtype=np.int64
        )
        cases.append((lengths, reused_prefix))

    # No reuse at all -> saved fraction must be exactly 0.
    lengths = rng.integers(8, 200, size=5)
    cases.append((lengths, np.zeros_like(lengths)))

    # Full reuse of every request -> saved fraction must be exactly 1.
    lengths = rng.integers(8, 200, size=5)
    cases.append((lengths, lengths.copy()))

    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for lengths, reused_prefix in _cases():
        ref = _oracle(lengths, reused_prefix)
        try:
            got = float(
                sol.prefill_flops_saved_fraction(lengths.copy(), reused_prefix.copy())
            )
        except Exception:
            return {"rel_err": 1.0}

        if abs(ref) > 1e-12:
            rel = abs(got - ref) / abs(ref)
        else:
            rel = abs(got - ref)
        worst = max(worst, rel)

    return {"rel_err": worst}
