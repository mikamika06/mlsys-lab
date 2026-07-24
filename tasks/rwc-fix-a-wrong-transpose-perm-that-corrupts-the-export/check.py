import itertools
import numpy as np


def _oracle_perm(x, ref):
    dims = tuple(range(x.ndim))
    for perm in itertools.permutations(dims):
        candidate = np.transpose(x, perm)
        if np.array_equal(candidate, ref):
            return perm
    raise AssertionError("no valid transpose permutation found")


def grade(sol, fx) -> dict:
    cases = [
        (np.arange(24).reshape(2, 3, 4), (1, 2, 0)),
        (np.arange(120).reshape(2, 3, 4, 5), (3, 1, 0, 2)),
        (np.arange(48).reshape(2, 2, 3, 4), (2, 0, 3, 1)),
        (np.arange(60).reshape(3, 4, 5), (2, 0, 1)),
    ]

    perm_ok = 1.0
    err = 0.0

    for x, true_perm in cases:
        exported = np.transpose(x, tuple(reversed(range(x.ndim))))
        reference = np.transpose(x, true_perm)

        try:
            got = tuple(sol.fix_transpose_perm(x, exported, reference))
        except Exception:
            perm_ok = 0.0
            err = float("inf")
            break

        oracle = _oracle_perm(x, reference)
        if got != oracle:
            perm_ok = 0.0

        try:
            repaired = np.transpose(x, got)
            err = max(err, float(np.max(np.abs(repaired.astype(np.float64) - reference.astype(np.float64)))))
        except Exception:
            err = float("inf")

    return {
        "perm_exact": perm_ok,
        "max_abs_err": err,
    }
