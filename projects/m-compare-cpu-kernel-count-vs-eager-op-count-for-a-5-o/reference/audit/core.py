import numpy as np


def analyze_pointwise_chain(trace):
    """Analyze 5-op pointwise chain kernel count vs eager op count."""
    eager = trace.get("eager_ops", 5)
    kernels = trace.get("cpu_kernels", 1)
    return {
        "eager_ops": eager,
        "cpu_kernels": kernels,
        "ratio": float(kernels) / float(eager)
    }


def find_autotuned_config(target_diff, candidates):
    """Find matching autotuned config ID for a recorded GPU reduction diff."""
    best_id = None
    min_err = float("inf")
    for cand in candidates:
        err = float(np.sum(np.abs(cand["output"] - target_diff)))
        if err < min_err:
            min_err = err
            best_id = cand["config_id"]
    return best_id
