import ref
import numpy as np


def check(workdir):
    from kvcache.ablation import measure_sink_ablation_blowup

    scenarios = ref.generate_scenarios()
    matched = 0
    for i, sc in enumerate(scenarios):
        try:
            err, ratio = measure_sink_ablation_blowup(sc["k"], sc["v"], sc["q"])
            if np.isclose(err, sc["expected_err"], atol=1e-5, rtol=1e-5) and \
               np.isclose(ratio, sc["expected_ratio"], atol=1e-5, rtol=1e-5):
                matched += 1
        except Exception:
            pass
    return {"ablation_matched": float(matched)}
