import numpy as np


def analyze_fallback(model_spec):
    x = model_spec["input"]
    unsupported = model_spec["unsupported_ops"]
    base_cost = float(np.prod(x.shape) * 1e-6 + 0.5)
    fallback_overhead = float(unsupported * 0.3)
    total_cost = base_cost + fallback_overhead
    fraction = fallback_overhead / (total_cost + 1e-9)
    return {"base_cost": base_cost, "fallback_overhead": fallback_overhead, "total_cost": total_cost, "fallback_fraction": fraction}
