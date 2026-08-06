import numpy as np


def _oracle_fold(W, b, gamma):
    return gamma.reshape(1, -1) * W, np.array(b, dtype=np.float64, copy=True)


def _rmsnorm(x, gamma, eps=1e-6):
    rms = np.sqrt(np.mean(x * x) + eps)
    return gamma * (x / rms)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(1234)
    max_err = 0.0

    cases = [
        (4, 3, 5),
        (8, 16, 2),
        (3, 7, 9),
    ]

    for m, d, n in cases:
        W = rng.normal(size=(m, d)).astype(np.float64)
        b = rng.normal(size=(m,)).astype(np.float64)
        gamma = rng.normal(size=(d,)).astype(np.float64)
        X = rng.normal(size=(n, d)).astype(np.float64)

        try:
            folded_W, folded_b = sol.fold_rmsnorm_gamma(W.tolist(), b.tolist(), gamma.tolist())
        except Exception:
            return {"max_abs_err": float("inf")}

        ref_W, ref_b = _oracle_fold(W, b, gamma)

        ref_y = np.stack([_rmsnorm(x, gamma) for x in X])
        ref_out = ref_y @ W.T + b

        folded_y = np.stack([
            x / np.sqrt(np.mean(x * x) + 1e-6)
            for x in X
        ])
        cand_out = folded_y @ np.asarray(folded_W).T + np.asarray(folded_b)

        err = max(
            float(np.max(np.abs(np.asarray(folded_W) - ref_W))),
            float(np.max(np.abs(np.asarray(folded_b) - ref_b))),
            float(np.max(np.abs(cand_out - ref_out))),
        )
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
