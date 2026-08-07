import math
import numpy as np


def _oracle_causal(logits):
    q = len(logits)
    k = len(logits[0]) if q > 0 else 0
    out = [[float(val) for val in row] for row in logits]
    for i in range(q):
        for j in range(k):
            if j > i:
                out[i][j] = -math.inf
    return out


def _oracle_alibi(logits, slope):
    q = len(logits)
    k = len(logits[0]) if q > 0 else 0
    out = [[float(val) for val in row] for row in logits]
    for i in range(q):
        for j in range(k):
            out[i][j] += float(slope) * (j - i)
    return out


def _max_abs(a, b):
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a_arr) & np.isfinite(b_arr)
    if np.any(a_arr[~finite] != b_arr[~finite]):
        return float("inf")
    if np.any(finite):
        return float(np.max(np.abs(a_arr[finite] - b_arr[finite])))
    return 0.0


def grade(sol, fx) -> dict:
    try:
        cases = [
            [[0.0, 1.0], [2.0, 3.0]],
            [[i / 7.0 for i in range(j, j + 4)] for j in range(0, 12, 4)],
            [[1.5, -2.0, 4.0], [0.0, 1.0, 2.0]],
        ]

        causal_err = 0.0
        alibi_err = 0.0

        for logits in cases:
            got_causal = sol.apply_attention_bias(logits, is_causal=True)
            ref_causal = _oracle_causal(logits)
            causal_err = max(causal_err, _max_abs(got_causal, ref_causal))

            got_alibi = sol.apply_attention_bias(logits, alibi_slope=0.25)
            ref_alibi = _oracle_alibi(logits, 0.25)
            alibi_err = max(alibi_err, _max_abs(got_alibi, ref_alibi))

            got_both = sol.apply_attention_bias(logits, is_causal=True, alibi_slope=0.25)
            ref_both = _oracle_causal(_oracle_alibi(logits, 0.25))
            causal_err = max(causal_err, _max_abs(got_both, ref_both))

        return {
            "max_abs_err": causal_err,
            "alibi_max_abs_err": alibi_err,
        }
    except Exception:
        return {
            "max_abs_err": float("inf"),
            "alibi_max_abs_err": float("inf"),
        }
