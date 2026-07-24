import numpy as np
from mlsys.scorers import rel_err

def _reference(batches, momentum):
    scale = np.zeros(batches[0].shape[1], dtype=np.float64)
    for batch in batches:
        absmax = np.max(np.abs(batch), axis=0)
        scale = momentum * scale + (1 - momentum) * absmax
    return scale

def grade(sol, fx):
    rng = np.random.default_rng(12345)
    num_batches = 12
    batch_size = 64
    num_tensors = 7
    batches = [rng.standard_normal((batch_size, num_tensors)) for _ in range(num_batches)]
    momentum = 0.9

    try:
        got = sol.moving_absmax(batches, momentum)
    except Exception:
        return {"rel_err": float("inf")}

    ref = _reference(batches, momentum)

    if not isinstance(got, np.ndarray) or got.dtype != np.float64:
        return {"rel_err": float("inf")}

    err = rel_err(ref, got)
    return {"rel_err": err}
