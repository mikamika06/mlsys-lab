import numpy as np


def _oracle_causal(logits):
    out = np.asarray(logits, dtype=np.float64).copy()
    q, k = out.shape
    mask = np.zeros((q, k), dtype=np.float64)
    mask[np.triu_indices(q, k, 1)] = -np.inf
    return out + mask


def _oracle_alibi(logits, slope):
    out = np.asarray(logits, dtype=np.float64).copy()
    q, k = out.shape
    q_idx = np.arange(q)[:, None]
    kv_idx = np.arange(k)[None, :]
    return out + slope * (kv_idx - q_idx)


def _max_abs(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if np.any(a[~finite] != b[~finite]):
        return float("inf")
    if np.any(finite):
        return float(np.max(np.abs(a[finite] - b[finite])))
    return 0.0


def grade(sol, fx) -> dict:
    try:
        cases = [
            np.array([[0.0, 1.0], [2.0, 3.0]]),
            np.arange(12, dtype=np.float64).reshape(3, 4) / 7.0,
            np.array([[1.5, -2.0, 4.0], [0.0, 1.0, 2.0]]),
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
