import numpy as np


def _softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def kv_grouping_mean_pool_kl(q, k, group_size):
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
