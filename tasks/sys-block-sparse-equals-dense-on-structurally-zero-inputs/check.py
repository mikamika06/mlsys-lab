import numpy as np


def _dense_oracle(Q, K, V, block_mask, block_size):
    n, d = Q.shape
    scores = (Q @ K.T) / np.sqrt(float(d))
    allowed = np.zeros((n, n), dtype=bool)
    blocks = block_mask.shape[0]
    for bi in range(blocks):
        for bj in range(blocks):
            if block_mask[bi, bj]:
                r0 = bi * block_size
                r1 = min(n, r0 + block_size)
                c0 = bj * block_size
                c1 = min(n, c0 + block_size)
                allowed[r0:r1, c0:c1] = True
    masked = np.where(allowed, scores, -np.inf)
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    exp = np.exp(shifted)
    weights = exp / np.sum(exp, axis=1, keepdims=True)
    return weights @ V


def _oracle_ratio(n, d, block_mask, block_size):
    active = int(np.sum(block_mask))
    dense = n * n * d
    sparse = active * block_size * block_size * d
    return float(dense / sparse)


def grade(sol, fx) -> dict:
    cases = [
        (8, 4, 2),
        (12, 3, 3),
        (16, 5, 4),
    ]
    max_err = 0.0
    ratio_ok = 1.0

    for n, d, b in cases:
        blocks = n // b
        rng = np.random.default_rng(n + d + b)
        Q = rng.normal(size=(n, d))
        K = rng.normal(size=(n, d))
        V = rng.normal(size=(n, d + 1))

        mask = np.zeros((blocks, blocks), dtype=bool)
        for i in range(blocks):
            mask[i, i] = True
            if i + 1 < blocks:
                mask[i, i + 1] = True

        K = K.copy()
        for bi in range(blocks):
            for bj in range(blocks):
                if not mask[bi, bj]:
                    K[bj * b:(bj + 1) * b] = 0.0

        try:
            got, ratio = sol.block_sparse_attention(Q, K, V, mask, b)
        except Exception:
            return {"max_abs_err": float("inf"), "savings_ratio": 0.0}

        ref = _dense_oracle(Q, K, V, mask, b)
        err = float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref)))
        max_err = max(max_err, err)

        if abs(float(ratio) - _oracle_ratio(n, d, mask, b)) > 1e-12:
            ratio_ok = 0.0

    return {
        "max_abs_err": max_err,
        "savings_ratio": ratio_ok,
    }
