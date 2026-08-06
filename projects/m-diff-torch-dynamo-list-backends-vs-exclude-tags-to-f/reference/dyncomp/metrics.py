import time
import torch
from dyncomp.noop import noop_backend


def measure_ratios(model: torch.nn.Module, example_inputs: list[torch.Tensor]) -> dict[str, float]:
    """Measure compile-time and run-time ratio of no-op backend vs inductor."""
    torch._dynamo.reset()
    start_compile = time.perf_counter()
    opt_noop = torch.compile(model, backend=noop_backend)
    _ = opt_noop(*example_inputs)
    noop_compile_time = time.perf_counter() - start_compile

    start_run = time.perf_counter()
    for _ in range(10):
        _ = opt_noop(*example_inputs)
    noop_run_time = (time.perf_counter() - start_run) / 10.0

    torch._dynamo.reset()
    start_compile = time.perf_counter()
    opt_inductor = torch.compile(model, backend="inductor")
    _ = opt_inductor(*example_inputs)
    inductor_compile_time = time.perf_counter() - start_compile

    start_run = time.perf_counter()
    for _ in range(10):
        _ = opt_inductor(*example_inputs)
    inductor_run_time = (time.perf_counter() - start_run) / 10.0

    compile_ratio = noop_compile_time / (inductor_compile_time + 1e-9)
    run_ratio = noop_run_time / (inductor_run_time + 1e-9)

    return {
        "compile_ratio": float(compile_ratio),
        "run_ratio": float(run_ratio),
    }
