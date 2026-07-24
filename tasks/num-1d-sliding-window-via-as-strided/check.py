"""Grader: zero-copy 1D sliding window built with as_strided.

Oracle = NumPy itself. The reference window matrix is materialised with
np.stack([x[i:i+w] ...]) at grade time; nothing is hardcoded.
"""
import numpy as np

from mlsys import scorers, probe  # noqa: F401


def _cases():
    rng = np.random.default_rng(0)
    out = []

    # plain contiguous float64
    out.append((rng.normal(size=17).astype(np.float64), 4))
    # contiguous float32 (different itemsize)
    out.append((rng.normal(size=12).astype(np.float32), 5))
    # contiguous int32 with an easily-readable pattern
    out.append((np.arange(9, dtype=np.int32), 3))
    # w == 1  -> (N, 1)
    out.append((rng.integers(-50, 50, size=8).astype(np.int64), 1))
    # non-contiguous view: stride is 3 * itemsize
    base = rng.normal(size=31).astype(np.float64)
    out.append((base[::3], 4))
    # non-contiguous view of a float32 buffer, reversed
    base32 = rng.normal(size=20).astype(np.float32)
    out.append((base32[::-2], 3))
    # w == N -> exactly one window
    x_full = rng.normal(size=6).astype(np.float64)
    out.append((x_full, 6))
    return out


def _reference(x, w):
    n = x.shape[0]
    return np.stack([x[i:i + w] for i in range(n - w + 1)])


def grade(sol, fx) -> dict:
    cases = _cases()

    total_bytes = 0
    same_bytes = 0
    shares = 0
    ok_cases = 0
    hard_fail = False

    for x, w in cases:
        ref = _reference(x, w)
        try:
            got = sol.sliding_window(x, w)
        except Exception:
            hard_fail = True
            total_bytes += ref.nbytes
            continue

        got_arr = np.asarray(got)
        total_bytes += ref.nbytes

        if got_arr.shape != ref.shape or got_arr.dtype != ref.dtype:
            continue

        frac = scorers.byte_exact_fraction(np.ascontiguousarray(got_arr),
                                           np.ascontiguousarray(ref))
        same_bytes += frac * ref.nbytes

        try:
            if np.shares_memory(got_arr, x):
                shares += 1
        except Exception:
            pass

        if frac >= 1.0:
            ok_cases += 1

    # ValueError contract on out-of-range window sizes
    guard_ok = 0
    guard_total = 0
    probe_x = np.arange(5, dtype=np.float64)
    for bad_w in (0, -1, 6):
        guard_total += 1
        try:
            sol.sliding_window(probe_x, bad_w)
        except ValueError:
            guard_ok += 1
        except Exception:
            pass

    if hard_fail or total_bytes == 0:
        return {
            "byte_exact_fraction": 0.0,
            "zero_copy_fraction": 0.0,
            "cases_ok": float(ok_cases),
        }

    byte_frac = same_bytes / total_bytes
    # the ValueError contract folds into the byte metric: missing it caps below 1.0
    if guard_ok < guard_total:
        byte_frac = min(byte_frac, 0.5)

    return {
        "byte_exact_fraction": float(byte_frac),
        "zero_copy_fraction": float(shares) / float(len(cases)),
        "cases_ok": float(ok_cases),
    }
