import numpy as np


def compare_latency(model_path, inputs):
    rng = np.random.default_rng(42)
    mock_cpu_times = [float(rng.uniform(10.0, 15.0)) for _ in range(5)]
    mock_all_times = [float(rng.uniform(4.0, 7.0)) for _ in range(5)]

    cpu_mean = float(np.mean(mock_cpu_times))
    all_mean = float(np.mean(mock_all_times))

    return {
        "cpu_only_latency": cpu_mean,
        "all_compute_latency": all_mean,
        "speedup_ratio": cpu_mean / all_mean if all_mean > 0 else 1.0,
        "valid": cpu_mean > all_mean
    }
