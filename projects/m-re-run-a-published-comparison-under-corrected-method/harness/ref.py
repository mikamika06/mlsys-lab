import numpy as np


def make_test_cases():
    np.random.seed(42)
    runs_matching = {"context_length": 4096}
    runs_mismatched_a = {"context_length": 4096}
    runs_mismatched_b = {"context_length": 8192}

    latencies = np.random.normal(loc=100.0, scale=10.0, size=30).tolist()

    return {
        "matching_a": runs_matching,
        "matching_b": runs_matching,
        "mismatch_a": runs_mismatched_a,
        "mismatch_b": runs_mismatched_b,
        "latencies": latencies
    }
