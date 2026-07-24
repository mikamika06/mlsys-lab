import numpy as np


def _oracle(W, nbits):
    W = np.asarray(W, dtype=np.float64)
    shape = W.shape
    blocks = W.reshape(*shape[:-1], -1, 4)
    abs_blocks = np.abs(blocks)

    order = np.argsort(abs_blocks, axis=-1)
    keep_mask = np.zeros_like(blocks, dtype=bool)
    np.put_along_axis(keep_mask, order[..., 2:], True, axis=-1)

    pruned = np.where(keep_mask, blocks, 0.0)
    survivor_abs = np.where(keep_mask, abs_blocks, 0.0)

    count = keep_mask.sum(axis=-1, keepdims=True).astype(np.float64)
    sum_abs = survivor_abs.sum(axis=-1, keepdims=True)
    scale = np.where(count > 0, sum_abs / np.maximum(count, 1.0), 1.0)

    qmax = 2 ** (nbits - 1) - 1
    code = np.clip(np.round(pruned / scale), -qmax, qmax)
    dequant = np.where(keep_mask, code * scale, 0.0)

    return dequant.reshape(shape)


def _cases():
    rng = np.random.default_rng(21)
    cases = []
    cases.append((rng.normal(size=(6, 16)), 4))
    cases.append((rng.normal(size=(3, 32)) * 3.0, 4))
    cases.append((rng.uniform(-5, 5, size=(4, 8)), 3))
    cases.append((rng.normal(loc=2.0, scale=0.5, size=(5, 24)), 8))
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for W, nbits in _cases():
        ref = _oracle(W, nbits)
        try:
            got = sol.compound_prune_quantize_2_4(np.array(W, copy=True), nbits)
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": worst}
