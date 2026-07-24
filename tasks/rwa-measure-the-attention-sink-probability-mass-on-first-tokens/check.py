import numpy as np


def _oracle_attention_sink_mass(logits, k):
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    attn = np.exp(x)
    attn = attn / np.sum(attn, axis=1, keepdims=True)
    col_mass = np.sum(attn, axis=0)
    return float(np.sum(col_mass[:k]) / np.sum(col_mass))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array(
                [
                    [4.0, 1.0, 0.0],
                    [3.0, 2.0, 0.0],
                    [2.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            1,
        ),
        (
            np.array(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [3.0, 0.0, 1.0, 2.0],
                    [2.0, 3.0, 0.0, 1.0],
                    [1.0, 2.0, 3.0, 0.0],
                ],
                dtype=np.float64,
            ),
            2,
        ),
        (
            np.array(
                [
                    [8.0, -2.0, -3.0, -4.0, -5.0],
                    [7.0, -1.0, -2.0, -3.0, -4.0],
                    [6.0, -2.0, -1.0, -2.0, -3.0],
                    [5.0, -3.0, -2.0, -1.0, -2.0],
                    [4.0, -4.0, -3.0, -2.0, -1.0],
                ],
                dtype=np.float64,
            ),
            1,
        ),
    ]

    worst = 0.0
    for logits, k in cases:
        try:
            got = float(sol.attention_sink_mass(logits, k))
        except Exception:
            return {"rel_err": 1.0}
        ref = _oracle_attention_sink_mass(logits, k)
        worst = max(worst, abs(got - ref) / (abs(ref) + 1e-12))

    return {"rel_err": worst}
