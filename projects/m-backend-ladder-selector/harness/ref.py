import sys
import types

def setup_mock_backends():
    faulty_mod = types.ModuleType("flashsel.backends.faulty")
    faulty_mod.is_available = lambda: (_ for _ in ()).throw(RuntimeError("fail"))
    sys.modules["flashsel.backends.faulty"] = faulty_mod

    ideal_mod = types.ModuleType("flashsel.backends.ideal")
    ideal_mod.is_available = lambda: True
    def compute(q, k, v):
        import numpy as np
        scale = 1.0 / np.sqrt(q.shape[-1])
        scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
        return np.matmul(exp_scores / sum_exp, v)
    ideal_mod.compute = compute
    sys.modules["flashsel.backends.ideal"] = ideal_mod
