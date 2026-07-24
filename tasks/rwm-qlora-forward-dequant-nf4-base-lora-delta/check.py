import numpy as np
from mlsys import scorers

NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=np.float64)


def _oracle_forward(x, codes, absmax, blocksize, A, B, alpha):
    d_out, d_in = codes.shape
    n_blocks = d_in // blocksize
    levels = NF4_LEVELS[codes].reshape(d_out, n_blocks, blocksize)
    w_dq = (levels * absmax[:, :, None]).reshape(d_out, d_in)
    r = A.shape[0]
    scaling = float(alpha) / r
    delta = scaling * (B @ A)
    w_eff = w_dq + delta
    return x @ w_eff.T


def grade(sol, fx) -> dict:
    """
    Random (x, nf4_codes, absmax, blocksize, A, B, alpha) combinations;
    compares the QLoRA forward output against a NumPy oracle that
    dequantizes NF4 blockwise and adds the scaled LoRA delta.
    """
    rng = np.random.default_rng(0)
    worst = 0.0

    for _ in range(6):
        blocksize = int(rng.choice([2, 4, 8]))
        n_blocks = int(rng.integers(2, 6))
        d_in = blocksize * n_blocks
        d_out = int(rng.integers(2, 8))
        r = int(rng.integers(1, 5))
        n = int(rng.integers(1, 6))

        codes = rng.integers(0, 16, size=(d_out, d_in)).astype(np.int64)
        absmax = rng.uniform(0.1, 3.0, size=(d_out, n_blocks)).astype(np.float64)
        A = rng.standard_normal((r, d_in)).astype(np.float64)
        B = rng.standard_normal((d_out, r)).astype(np.float64)
        alpha = float(rng.uniform(0.5, 8.0))
        x = rng.standard_normal((n, d_in)).astype(np.float64)

        expected = _oracle_forward(x, codes, absmax, blocksize, A, B, alpha)
        try:
            got = np.asarray(
                sol.qlora_forward(x.copy(), codes.copy(), absmax.copy(), blocksize,
                                   A.copy(), B.copy(), alpha),
                dtype=np.float64,
            )
            if got.shape != expected.shape:
                return {"rel_err": float("inf")}
            err = scorers.rel_err(expected, got)
        except Exception:
            return {"rel_err": float("inf")}

        worst = max(worst, err)

    return {"rel_err": worst}
