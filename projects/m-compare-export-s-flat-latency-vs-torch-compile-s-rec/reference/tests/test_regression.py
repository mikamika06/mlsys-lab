import numpy as np
from exportbench.runner import SimulatedModule, benchmark_runtimes


def test_dynamic_shape_compliance():
    model = SimulatedModule(hidden_dim=64, static_compile=False)
    batches = [1, 2, 4, 8, 16, 32, 1, 2, 4]
    results = benchmark_runtimes(model, batches)

    compile_spikes = sum(1 for lat in results["compile_latencies"] if lat > 0.01)
    assert compile_spikes <= 1, f"Expected at most 1 compilation spike, got {compile_spikes}"
