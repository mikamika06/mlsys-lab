import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _full_recompute(x, Wq, Wk, Wv):
    outputs = []
    scale = np.sqrt(Wq.shape[1])
    for t in range(x.shape[0]):
        prefix = x[: t + 1]
        q = prefix[-1:] @ Wq
        k = prefix @ Wk
        v = prefix @ Wv
        scores = (q @ k.T) / scale
        weights = _softmax(scores)
        outputs.append(weights @ v)
    return np.concatenate(outputs, axis=0)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(6, 4))
    Wq = rng.normal(size=(4, 4))
    Wk = rng.normal(size=(4, 4))
    Wv = rng.normal(size=(4, 4))

    ref = _full_recompute(x, Wq, Wk, Wv)

    try:
        got = np.asarray(sol.decode_steps(x, Wq, Wk, Wv), dtype=np.float64)
        err = float(np.max(np.abs(got - ref)))
    except Exception:
        err = float("inf")

    return {"max_abs_err": err}
