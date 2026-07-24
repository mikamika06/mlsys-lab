import sys
import numpy as np

def _size_of(obj):
    if isinstance(obj, np.ndarray):
        return obj.nbytes + sys.getsizeof(obj)
    elif isinstance(obj, dict):
        size = sys.getsizeof(obj)
        for k, v in obj.items():
            size += sys.getsizeof(k) + _size_of(v)
        return size
    elif isinstance(obj, list):
        size = sys.getsizeof(obj)
        for item in obj:
            size += _size_of(item)
        return size
    else:
        return sys.getsizeof(obj)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    params = {"w": rng.standard_normal((5, 7), dtype=np.float32),
              "b": rng.standard_normal((7,), dtype=np.float32)}
    grads = {k: rng.standard_normal(v.shape, dtype=v.dtype) for k, v in params.items()}
    optimizer_state = {"momentum_w": rng.standard_normal(params["w"].shape, dtype=params["w"].dtype),
                       "velocity_b": rng.standard_normal(params["b"].shape, dtype=params["b"].dtype)}
    activations = [rng.standard_normal((3, 5), dtype=np.float32) for _ in range(4)]

    ref_bytes = (_size_of(params) + _size_of(grads)
                 + _size_of(optimizer_state) + _size_of(activations))

    try:
        cand_bytes = sol.total_training_memory(params, grads, optimizer_state, activations)
    except Exception:
        return {"size_ratio": 0.0}

    ratio = float(ref_bytes) / float(cand_bytes)
    return {"size_ratio": ratio}
