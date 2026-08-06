CASES = [
    {"latency_ms": 1.5, "total_flops": 4.5e11, "peak_tflops_s": 300.0},
    {"latency_ms": 2.0, "total_flops": 1.2e12, "peak_tflops_s": 400.0},
    {"latency_ms": 0.8, "total_flops": 2.4e11, "peak_tflops_s": 250.0},
]

PUBLISHED_CASES = [
    {"total_flops": 5.0e11, "latency_ms": 2.5},
    {"total_flops": 1.0e12, "latency_ms": 4.0},
    {"total_flops": 2.0e12, "latency_ms": 5.0},
]

AUDIT_CASES = [
    {"base": 100.0, "opt": 25.0, "claimed": 4.0},
    {"base": 100.0, "opt": 50.0, "claimed": 3.0},
    {"base": 200.0, "opt": 40.0, "claimed": 5.0},
]
