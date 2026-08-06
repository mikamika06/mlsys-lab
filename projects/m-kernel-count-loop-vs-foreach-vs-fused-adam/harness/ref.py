import numpy as np


def generate_synthetic_params():
    np.random.seed(1337)
    devices = ["cuda:0", "cuda:1"]
    dtypes = ["float32", "float16"]

    params = []
    for i in range(20):
        dev = devices[i % len(devices)]
        dt = dtypes[(i // 2) % len(dtypes)]
        shape = (10, 10)
        params.append({
            "id": i,
            "device": dev,
            "dtype": dt,
            "param": np.random.randn(*shape).astype(np.float64),
            "grad": np.random.randn(*shape).astype(np.float64),
        })
    return params


def generate_states(params):
    states = []
    for p in params:
        states.append({
            "exp_avg": np.zeros_like(p["param"]),
            "exp_avg_sq": np.zeros_like(p["param"]),
            "step": 0
        })
    return states


def reference_grouping(params):
    groups = {}
    for p in params:
        key = (p.get("device", "cpu"), p.get("dtype", "float32"))
        if key not in groups:
            groups[key] = []
        groups[key].append(p)
    return groups


def reference_kernel_counts(params, num_steps=1):
    n_params = len(params)
    groups = reference_grouping(params)
    n_groups = len(groups)
    return {
        "loop": n_params * 4 * num_steps,
        "foreach": n_groups * 4 * num_steps,
        "fused": n_groups * 1 * num_steps,
    }
