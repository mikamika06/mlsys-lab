import numpy as np


def _full_batch_grad(X, y, w):
    r = X @ w - y
    N = X.shape[0]
    return (2.0 / N) * (X.T @ r)


def _gen_case(rng, unequal):
    D = int(rng.integers(2, 6))
    n_micro = int(rng.integers(2, 5))
    if unequal:
        sizes = [int(rng.integers(1, 9)) for _ in range(n_micro)]
    else:
        b = int(rng.integers(2, 6))
        sizes = [b] * n_micro
    micro_batches = []
    for b in sizes:
        X_i = rng.standard_normal((b, D))
        y_i = rng.standard_normal(b)
        micro_batches.append((X_i, y_i))
    w = rng.standard_normal(D)
    return micro_batches, w


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    worst = 0.0
    for i in range(8):
        micro_batches, w = _gen_case(rng, unequal=(i % 2 == 1))
        X_full = np.concatenate([mb[0] for mb in micro_batches], axis=0)
        y_full = np.concatenate([mb[1] for mb in micro_batches], axis=0)
        expected = _full_batch_grad(X_full, y_full, w)
        try:
            mb_copy = [(X_i.copy(), y_i.copy()) for X_i, y_i in micro_batches]
            got = np.asarray(sol.accumulate_grad(mb_copy, w.copy()), dtype=np.float64)
            if got.shape != expected.shape:
                worst = float("inf")
                break
            err = float(np.max(np.abs(got - expected)))
        except Exception:
            worst = float("inf")
            break
        worst = max(worst, err)
    return {"max_abs_err": worst}
