def check(workdir):
    from bakeoff.models import StackModel
    import numpy as np
    m = {"structural_equivalence": 0.0}
    try:
        cfg = {"dim": 32}
        model_a = StackModel(cfg)
        model_b = StackModel(cfg)
        x = np.ones((2, 32), dtype=np.float32)
        out_a = model_a.forward(x)
        out_b = model_b.forward(x)
        if out_a.shape == out_b.shape and np.allclose(out_a, out_b):
            m["structural_equivalence"] = 1.0
    except Exception:
        pass
    return m
