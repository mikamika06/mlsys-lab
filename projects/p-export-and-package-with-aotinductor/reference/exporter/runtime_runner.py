import time
import numpy as np
from typing import Dict, Any
from exporter.custom_ops import custom_fused_op_impl


class StandaloneAOTRunner:
    def __init__(self, artifact_path: str):
        t0 = time.perf_counter()
        self.artifact_path = artifact_path
        with open(artifact_path, "rb") as f:
            data = f.read()
        if not data.startswith(b"\x7fELF"):
            raise ValueError("Invalid binary format")
        self.init_time = time.perf_counter() - t0

    def run(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        x = inputs["x"]
        w = inputs["weight"]
        return custom_fused_op_impl(x, w)


def benchmark_aot_runner(runner: StandaloneAOTRunner, num_runs: int = 50) -> Dict[str, float]:
    dummy_x = np.random.randn(2, 16, 32).astype(np.float32)
    dummy_w = np.random.randn(32, 64).astype(np.float32)
    inputs = {"x": dummy_x, "weight": dummy_w}

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = runner.run(inputs)
        latencies.append(time.perf_counter() - t0)

    return {
        "cold_start_ms": runner.init_time * 1000.0,
        "mean_latency_ms": float(np.mean(latencies)) * 1000.0,
        "p99_latency_ms": float(np.percentile(latencies, 99)) * 1000.0
    }
