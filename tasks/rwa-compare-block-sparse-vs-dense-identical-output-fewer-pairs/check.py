import numpy as np


def _dense_oracle(Q, K, V, mask):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    scores = Q @ K.T / np.sqrt(Q.shape[1])
    out = np.zeros((Q.shape[0], V.shape[1]), dtype=np.float64)

    for i in range(Q.shape[0]):
        allowed = mask[i]
        if np.any(allowed):
            x = scores[i, allowed]
            x = x - np.max(x)
            p = np.exp(x)
            p = p / np.sum(p)
            out[i] = p @ V[allowed]

    return out


def _expected_pairs(mask, block_size):
    n = mask.shape[0]
    empty = 0
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            if not np.any(mask[i:i + block_size, j:j + block_size]):
                empty += 1
    return n * n - empty * block_size * block_size


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = []
    for n, d, m, b in [(8, 3, 2, 2), (8, 4, 3, 4), (6, 5, 2, 2)]:
        Q = rng.normal(size=(n, d))
        K = rng.normal(size=(n, d))
        V = rng.normal(size=(n, m))
        mask = rng.random((n, n)) > 0.55
        mask[:b, :b] = False
        mask[-b:, -b:] = False
        cases.append((Q, K, V, mask, b))

    max_err = 0.0
    pairs_ok = 1.0

    for Q, K, V, mask, b in cases:
        ref = _dense_oracle(Q, K, V, mask)
        try:
            got, count = sol.block_sparse_attention(Q, K, V, mask, b)
        except Exception:
            return {"max_abs_err": float("inf"), "attended_pairs": 0.0}

        got = np.asarray(got, dtype=np.float64)
        max_err = max(max_err, float(np.max(np.abs(got - ref))))
        if int(count) != _expected_pairs(mask, b):
            pairs_ok = 0.0

    return {
        "max_abs_err": max_err,
        "attended_pairs": pairs_ok,
    }
