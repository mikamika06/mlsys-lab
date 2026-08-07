import ref
import numpy as np

def check(workdir):
    from longctx.perplexity import measure_perplexity
    out = {"perplexity_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        logits = np.random.randn(32, 64)
        targets = np.random.randint(0, 64, size=(32,))
        want = ref.measure_perplexity(logits, targets, cfg["base_scale"])
        try:
            got = measure_perplexity(logits, targets, cfg["base_scale"])
            if np.isclose(got, want, rtol=1e-5, atol=1e-5):
                ok += 1
        except Exception:
            pass
    out["perplexity_matched"] = float(ok)
    return out
