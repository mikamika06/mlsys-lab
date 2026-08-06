import numpy as np

def generate_fixtures():
    np.random.seed(42)
    data = np.random.randn(1000, 64).astype(np.float32) + 100.0
    benchmarks = [
        {"kernel": "unfused", "bytes_transferred": 1.5e9, "time_ms": 1.2},
        {"kernel": "fused", "bytes_transferred": 0.8e9, "time_ms": 0.5}
    ]
    return data, benchmarks

def compute_fp16_variance_reference(x):
    x_fp16 = x.astype(np.float16).astype(np.float32)
    mean_sq = np.mean(x_fp16 ** 2, axis=-1)
    sq_mean = np.mean(x_fp16, axis=-1) ** 2
    single_pass = mean_sq - sq_mean
    two_pass = np.mean((x_fp16 - np.mean(x_fp16, axis=-1, keepdims=True)) ** 2, axis=-1)
    rel_err = np.mean(np.abs(single_pass - two_pass) / (np.abs(two_pass) + 1e-5))
    return float(rel_err)

def analyze_bandwidth_reference(records):
    unfused_bw = sum(r["bytes_transferred"] / (r["time_ms"] * 1e-3) for r in records if r["kernel"] == "unfused")
    fused_bw = sum(r["bytes_transferred"] / (r["time_ms"] * 1e-3) for r in records if r["kernel"] == "fused")
    return float(fused_bw / (unfused_bw + 1e-9) if unfused_bw > 0 else 1.5)
