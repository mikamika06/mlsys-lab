import numpy as np


def _oracle(x, W, p, seed, n_pre, n_between):
    """True forward value AND true recomputed value for a checkpointed
    linear -> relu -> dropout block, using one shared RNG stream.

    A correct checkpoint snapshots the RNG state right before drawing its
    own dropout mask and restores that snapshot before recomputing, so the
    recompute reproduces the identical mask (and therefore identical
    output) despite `n_between` unrelated draws from the same shared
    stream happening in between (the rest of the network running between
    the forward call and the later recompute-for-backward call).
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_pre):
        rng.random(3)  # unrelated earlier network parts sharing the RNG

    entry_state = rng.bit_generator.state
    mask = (rng.random(x.shape[0] * W.shape[1]).reshape(x.shape[0], W.shape[1]) >= p).astype(np.float64)
    h = np.maximum(x @ W, 0.0)
    y_forward = h * mask / (1.0 - p)

    for _ in range(n_between):
        rng.random(3)  # unrelated later network parts sharing the RNG

    rng.bit_generator.state = entry_state
    mask2 = (rng.random(x.shape[0] * W.shape[1]).reshape(x.shape[0], W.shape[1]) >= p).astype(np.float64)
    h2 = np.maximum(x @ W, 0.0)
    y_recomputed = h2 * mask2 / (1.0 - p)

    return y_forward, y_recomputed


def _cases():
    return [
        dict(seed=1, n_pre=0, n_between=3, p=0.3, data_seed=10),
        dict(seed=42, n_pre=2, n_between=1, p=0.5, data_seed=11),
        dict(seed=99, n_pre=5, n_between=7, p=0.2, data_seed=12),
    ]


def grade(sol, fx) -> dict:
    n, d, k = 5, 4, 6
    worst = 0.0
    for c in _cases():
        rng = np.random.default_rng(c["data_seed"])
        x = rng.standard_normal((n, d))
        W = rng.standard_normal((d, k))

        y_fwd_ref, y_recomp_ref = _oracle(x, W, c["p"], c["seed"], c["n_pre"], c["n_between"])

        try:
            out = sol.checkpointed_dropout_block(
                x.copy(), W.copy(), c["p"], c["seed"], c["n_pre"], c["n_between"]
            )
            y_fwd_got, y_recomp_got = out
            y_fwd_got = np.asarray(y_fwd_got, dtype=np.float64)
            y_recomp_got = np.asarray(y_recomp_got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if y_fwd_got.shape != y_fwd_ref.shape or y_recomp_got.shape != y_recomp_ref.shape:
            return {"max_abs_err": float("inf")}
        if not (np.all(np.isfinite(y_fwd_got)) and np.all(np.isfinite(y_recomp_got))):
            return {"max_abs_err": float("inf")}

        err = max(
            float(np.max(np.abs(y_fwd_got - y_fwd_ref))),
            float(np.max(np.abs(y_recomp_got - y_recomp_ref))),
        )
        worst = max(worst, err)

    return {"max_abs_err": worst}
