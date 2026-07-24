import numpy as np
from mlsys.scorers import channel_rel_err

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((32, 64))
    try:
        per_tensor, per_channel = sol.compare_quantization(W)
    except Exception:
        return {"channel_rel_err": float("inf")}
    if not isinstance(per_tensor, np.ndarray) or not isinstance(per_channel, np.ndarray):
        return {"channel_rel_err": float("inf")}
    if per_tensor.shape != W.shape or per_channel.shape != W.shape:
        return {"channel_rel_err": float("inf")}
    err = channel_rel_err(W, per_channel)
    return {"channel_rel_err": err}
