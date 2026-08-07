import numpy as np
from reference.kvcache.ablation import measure_sink_ablation_blowup
from reference.kvcache.mask import reconstruct_kept_mask


def generate_scenarios():
    rng = np.random.default_rng(123)
    scenarios = []
    for _ in range(3):
        k = rng.standard_normal((1, 2, 16, 32))
        v = rng.standard_normal((1, 2, 16, 32))
        q = rng.standard_normal((1, 2, 1, 32))
        err, ratio = measure_sink_ablation_blowup(k, v, q)
        scenarios.append({"k": k, "v": v, "q": q, "expected_err": err, "expected_ratio": ratio})
    return scenarios


def generate_mask_scenarios():
    scenarios = [
        {"dump": {"indices": [0, 1, 4, 7]}, "length": 10},
        {"dump": {"timestamps": np.array([1.5, 0.2, 2.1, 0.1, 1.9]), "threshold": 1.0}, "length": 5},
        {"dump": {}, "length": 6}
    ]
    return scenarios
