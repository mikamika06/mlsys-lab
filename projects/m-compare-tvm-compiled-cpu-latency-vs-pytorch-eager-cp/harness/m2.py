import time
import ref


def check(workdir):
    from tvm_bench.frontend import import_and_compile
    from tvm_bench.latency import (
        compute_latency_ratio,
        measure_eager_latency,
        measure_tvm_latency,
    )

    out = {"latency_ratio": 0.0, "timing_accurate": 0.0}

    model = ref.MockModel(["conv2d", "relu"], complexity=20000)
    compiled = import_and_compile(model, [1.0], ref.SUPPORTED_OPS)

    eager_ms = measure_eager_latency(model, [1.0], warmup=2, runs=5)
    tvm_ms = measure_tvm_latency(compiled, [1.0], warmup=2, runs=5)

    if eager_ms > 0 and tvm_ms > 0:
        out["timing_accurate"] = 1.0

    ratio = compute_latency_ratio(model, compiled, [1.0], warmup=2, runs=5)
    out["latency_ratio"] = float(ratio)

    return out
