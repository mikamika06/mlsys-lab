import numpy as np


def _oracle_expand(kv, num_query_heads):
    kv_arr = np.asarray(kv, dtype=np.float32)
    repeat = num_query_heads // kv_arr.shape[1]
    return np.repeat(kv_arr, repeat, axis=1).astype(np.float32)


def grade(sol, fx) -> dict:
    cases = [
        (
            [
                [
                    [[0.5, -1.0, 2.0]],
                    [[3.0, 4.0, -2.0]],
                ]
            ],
            4,
        ),
        (
            np.arange(2 * 4 * 3 * 2, dtype=np.float32).reshape(2, 4, 3, 2).tolist(),
            8,
        ),
        (
            np.array(
                np.random.default_rng(7).normal(size=(3, 3, 5, 4)),
                dtype=np.float32,
            ).tolist(),
            6,
        ),
    ]

    worst = 0.0
    for kv, nq in cases:
        ref = _oracle_expand(kv, nq)
        try:
            got = sol.expand_gqa_kv(kv, nq)
        except Exception:
            return {"max_abs_err": float("inf")}
        got_arr = np.asarray(got, dtype=np.float32)
        if got_arr.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got_arr - ref))))
    return {"max_abs_err": worst}
