import numpy as np


def run_with_options(model_spec, compute_units="All"):
    x = model_spec["input"]
    out = np.tanh(x) * 2.0 - 1.0
    if compute_units == "CPUOnly":
        latency = float(np.prod(x.shape) * 1e-6 + 2.0)
    else:
        latency = float(np.prod(x.shape) * 1e-6 + 0.5 + model_spec["unsupported_ops"] * 0.3)
    return {"output": out, "latency": latency, "compute_units": compute_units}
