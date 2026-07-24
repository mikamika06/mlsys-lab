import numpy as np


def _bf16_round(x):
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    lsb = (u >> 16) & 1
    rounded = u + np.uint32(0x7FFF) + lsb
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32).astype(np.float64)


def _ref_online_softmax(x, B):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    xb = _bf16_round(x)

    m = np.float64(-np.inf)
    l = np.float64(0.0)

    for start in range(0, n, B):
        block = _bf16_round(xb[start:start + B])
        block_max = _bf16_round(np.max(block))
        new_m = _bf16_round(max(m, block_max))

        if np.isneginf(m):
            old_term = 0.0
        else:
            old_term = _bf16_round(np.exp(_bf16_round(m - new_m)))

        old_scaled = _bf16_round(l * old_term)
        shifted = _bf16_round(block - new_m)
        exp_block = _bf16_round(np.exp(shifted))
        block_sum = _bf16_round(np.sum(exp_block))
        l = _bf16_round(old_scaled + block_sum)
        m = new_m

    out = np.empty(n, dtype=np.float64)
    denom = _bf16_round(l)
    for i in range(n):
        num = _bf16_round(np.exp(_bf16_round(xb[i] - m)))
        out[i] = _bf16_round(num / denom)
    return out


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        (rng.normal(size=37) * 3.0),
        (rng.normal(size=129) * 8.0),
        (np.linspace(-10, 10, 257)),
        (rng.normal(size=513) * 1.5),
    ]

    max_err = 0.0
    for x in cases:
        for B in (16, 64, 256):
            try:
                got = np.asarray(sol.tiled_online_softmax(x.copy(), B), dtype=np.float64)
            except Exception:
                return {"max_abs_err": float("inf")}
            ref = _ref_online_softmax(x, B)
            max_err = max(max_err, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": max_err}
