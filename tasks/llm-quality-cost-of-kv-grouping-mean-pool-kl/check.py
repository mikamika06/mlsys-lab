import numpy as np


def _softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _oracle(q, k, group_size):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)

    b, h, tq, d = q.shape

    full_logits = np.einsum("bhid,bhjd->bhij", q, k) / np.sqrt(d)
    full = _softmax(full_logits, axis=-1)

    groups = h // group_size
    pooled = k.reshape(
        b, groups, group_size, k.shape[2], d
    ).mean(axis=2)

    grouped = np.empty_like(full)
    for head in range(h):
        logits = np.einsum(
            "bid,bjd->bij",
            q[:, head],
            pooled[:, head // group_size],
        ) / np.sqrt(d)
        grouped[:, head] = _softmax(logits, axis=-1)

    kl = np.sum(
        full * (np.log(full + 1e-12) - np.log(grouped + 1e-12)),
        axis=-1,
    )
    return float(np.mean(kl))


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = [
        (rng.normal(size=(1, 4, 3, 8)), rng.normal(size=(1, 4, 5, 8)), 2),
        (rng.normal(size=(2, 8, 4, 6)), rng.normal(size=(2, 8, 7, 6)), 4),
        (rng.normal(size=(1, 6, 2, 5)), rng.normal(size=(1, 6, 3, 5)), 3),
    ]

    max_error = 0.0
    for q, k, group_size in cases:
        expected = _oracle(q, k, group_size)
        try:
            got = float(sol.kv_grouping_mean_pool_kl(q, k, group_size))
        except Exception:
            return {"mean_kl": float("inf")}
        max_error = max(max_error, abs(got - expected))

    return {"mean_kl": max_error}
