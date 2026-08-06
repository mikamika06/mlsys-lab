import numpy as np


def generate_fixtures():
    np.random.seed(42)
    cases = []
    for i in range(5):
        eager_ops = 5
        cpu_kernels = int(np.random.choice([1, 2, 3]))
        ratio = float(cpu_kernels) / float(eager_ops)
        cases.append({
            "trace_id": i,
            "eager_ops": eager_ops,
            "cpu_kernels": cpu_kernels,
            "ratio": ratio,
            "ir_lines": [f"op_{j}" for j in range(eager_ops)]
        })

    diff_cases = []
    for i in range(5):
        true_cfg_id = int(np.random.randint(0, 4))
        base = np.random.randn(16, 16).astype(np.float32)
        candidates = []
        target_diff = None
        for c_id in range(4):
            noise = np.random.randn(16, 16).astype(np.float32) * (0.1 * (c_id + 1))
            res = base + noise
            candidates.append({"config_id": c_id, "output": res})
            if c_id == true_cfg_id:
                target_diff = res
        diff_cases.append({
            "case_id": i,
            "target_diff": target_diff,
            "candidates": candidates,
            "expected_config_id": true_cfg_id
        })
    return cases, diff_cases


def analyze_pointwise_chain(trace):
    eager = trace["eager_ops"]
    kernels = trace["cpu_kernels"]
    return {"eager_ops": eager, "cpu_kernels": kernels, "ratio": float(kernels) / float(eager)}


def find_autotuned_config(target_diff, candidates):
    best_id = None
    min_err = float("inf")
    for cand in candidates:
        err = float(np.sum(np.abs(cand["output"] - target_diff)))
        if err < min_err:
            min_err = err
            best_id = cand["config_id"]
    return best_id
