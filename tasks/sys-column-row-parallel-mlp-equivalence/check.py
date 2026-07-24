import numpy as np


def _gelu(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _oracle(x, w1_shards, b1_shards, w2_shards, b2):
    x = np.asarray(x, dtype=np.float64)
    W1 = np.concatenate([np.asarray(w, dtype=np.float64) for w in w1_shards], axis=1)
    b1 = np.concatenate([np.asarray(b, dtype=np.float64) for b in b1_shards])
    W2 = np.concatenate([np.asarray(w, dtype=np.float64) for w in w2_shards], axis=0)
    b2 = np.asarray(b2, dtype=np.float64)
    a = _gelu(x @ W1 + b1)
    return a @ W2 + b2


def _make_case(rng, m, d, hs, d_out):
    x = rng.standard_normal((m, d))
    w1_shards = [rng.standard_normal((d, h)) * 0.2 for h in hs]
    b1_shards = [rng.standard_normal(h) * 0.1 for h in hs]
    w2_shards = [rng.standard_normal((h, d_out)) * 0.2 for h in hs]
    b2 = rng.standard_normal(d_out) * 0.1
    return x, w1_shards, b1_shards, w2_shards, b2


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        (5, 8, [3, 5, 4], 6),
        (3, 4, [2, 2], 3),
        (7, 6, [1, 3, 2, 5], 4),
        (1, 3, [4], 2),
    ]

    worst = 0.0
    for m, d, hs, d_out in cases:
        x, w1_shards, b1_shards, w2_shards, b2 = _make_case(rng, m, d, hs, d_out)
        ref = _oracle(x, w1_shards, b1_shards, w2_shards, b2)
        try:
            got = np.asarray(
                sol.mlp_tensor_parallel(x, w1_shards, b1_shards, w2_shards, b2),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        err = float(np.max(np.abs(got - ref)))
        worst = max(worst, err)
    return {"max_abs_err": worst}
