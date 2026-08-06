import numpy as np
from flashsel.selector import select_backend

def _ref_attention(q, k, v):
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    attn = exp_scores / sum_exp
    return np.matmul(attn, v)

def execute_with_fallback(ladder, q, k, v, preferences=None):
    backend = select_backend(ladder, preferences)
    if backend == "flashsel.backends.ideal":
        mod = __import__(backend, fromlist=["compute"])
        if hasattr(mod, "compute"):
            return mod.compute(q, k, v)
    return _ref_attention(q, k, v)
