import ref
import numpy as np

def check(workdir):
    from tflite_pipe.converter import verify_dual_entry
    out = {"numerics_matched": 0.0}
    ok = 0
    rng = np.random.RandomState(42)
    for i, cfg in enumerate(ref.CONFIGS):
        dummy_input = rng.randn(2, 4).astype(np.float32)
        want = ref.verify_dual_entry(cfg, dummy_input)
        got = verify_dual_entry(cfg, dummy_input)
        if got is not None and np.allclose(got, want, atol=1e-5):
            ok += 1
    out["numerics_matched"] = float(ok)
    return out
