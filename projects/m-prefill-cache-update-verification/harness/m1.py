import ref
import numpy as np


def check(workdir):
    from cacheverify.verify import verify_prefill_update
    rng = np.random.default_rng(42)
    outputs = [rng.normal(size=(1, 8, 16, 64)) for _ in range(2)]
    ref_cache = [o + rng.normal(scale=1e-7, size=o.shape) for o in outputs]
    got = verify_prefill_update(outputs, ref_cache)
    want = ref.verify_prefill_update(outputs, ref_cache)
    return {"max_abs_err": float(abs(got - want))}
