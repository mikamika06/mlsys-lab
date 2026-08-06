import numpy as np
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from bfacc.repro import deterministic_sum, diagnose_loss_spike

    out = {"bit_exact_matches": 0.0, "spike_diagnosed": 0.0}
    
    rng = np.random.RandomState(99)
    chunks1 = [rng.randn(100, 100).astype(np.float32) for _ in range(10)]
    chunks2 = [chunks1[i] for i in [4, 0, 8, 2, 1, 7, 3, 9, 5, 6]]
    
    res1 = deterministic_sum(chunks1, axis=0)
    res2 = deterministic_sum(chunks2, axis=0)
    
    if np.array_equal(res1, res2) and res1.dtype == np.float32:
        out["bit_exact_matches"] = 1.0
        
    history = ref.generate_loss_series(seed=2026)
    diag = diagnose_loss_spike(history, threshold=2.0)
    
    if isinstance(diag, dict) and diag.get("spiked") is True and diag.get("step") == 25:
        out["spike_diagnosed"] = 1.0
        
    return out
