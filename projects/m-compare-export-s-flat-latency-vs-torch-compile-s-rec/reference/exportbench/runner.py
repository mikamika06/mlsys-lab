import time
import numpy as np


class SimulatedModule:
    """Simulates PyTorch execution with compile vs export performance profiles."""
    def __init__(self, hidden_dim=256, static_compile=True):
        self.hidden_dim = hidden_dim
        self.static_compile = static_compile
        self.compiled_shapes = set()

    def run_compiled(self, x):
        batch_size = x.shape[0]
        if self.static_compile:
            if batch_size not in self.compiled_shapes:
                self.compiled_shapes.add(batch_size)
                time.sleep(0.02)
        else:
            if len(self.compiled_shapes) == 0:
                self.compiled_shapes.add(-1)
                time.sleep(0.02)

        start = time.perf_counter()
        _ = np.dot(x, np.ones((self.hidden_dim, self.hidden_dim)))
        duration = time.perf_counter() - start
        return duration

    def run_exported(self, x):
        start = time.perf_counter()
        _ = np.dot(x, np.ones((self.hidden_dim, self.hidden_dim)))
        duration = time.perf_counter() - start
        return duration


def benchmark_runtimes(model, batch_sequence):
    compile_latencies = []
    export_latencies = []

    for batch_size in batch_sequence:
        x = np.random.randn(batch_size, model.hidden_dim)
        c_lat = model.run_compiled(x)
        e_lat = model.run_exported(x)
        compile_latencies.append(c_lat)
        export_latencies.append(e_lat)

    return {
        "compile_latencies": compile_latencies,
        "export_latencies": export_latencies,
        "max_compile_spike": max(compile_latencies),
        "max_export_spike": max(export_latencies),
        "compile_std": float(np.std(compile_latencies)),
        "export_std": float(np.std(export_latencies)),
    }
