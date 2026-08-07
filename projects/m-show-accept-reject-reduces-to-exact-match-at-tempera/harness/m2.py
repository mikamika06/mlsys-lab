import ref
import numpy as np

def check(workdir):
    from specsampling.core import measure_acceptance_rates
    out = {"rates_matched": 0.0}
    np.random.seed(123)
    t_logits = np.random.randn(20)
    d_logits = np.random.randn(20)
    temps = [0.0, 0.5, 1.0, 1.5]
    want = ref.measure_acceptance_rates(t_logits, d_logits, temps)
    try:
        got = measure_acceptance_rates(t_logits, d_logits, temps)
        if len(got) == len(want) and all(abs(a - b) < 0.15 for a, b in zip(got, want)):
            out["rates_matched"] = 1.0
    except Exception:
        pass
    return out
