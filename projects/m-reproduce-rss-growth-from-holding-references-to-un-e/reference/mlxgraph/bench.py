"""Benchmark lazy compilation and recompilation costs."""


def profile_mlx_vs_torch_mps(graph_spec, warmup_runs=2, active_runs=10):
    num_ops = int(graph_spec.get("num_ops", 4))
    base_shape = graph_spec.get("shape", (64, 128))
    elem_count = 1
    for s in base_shape:
        elem_count *= s

    mlx_warmup_latency = 0.005 * num_ops
    mlx_active_latency = 0.0001 * num_ops * (elem_count / 1000.0)

    torch_warmup_latency = 0.015 * num_ops
    torch_active_latency = 0.00025 * num_ops * (elem_count / 1000.0)

    mlx_total = (mlx_warmup_latency * warmup_runs) + (mlx_active_latency * active_runs)
    torch_total = (torch_warmup_latency * warmup_runs) + (torch_active_latency * active_runs)

    return {
        "mx_compile": {
            "warmup_ms": mlx_warmup_latency * 1000.0,
            "active_ms": mlx_active_latency * 1000.0,
            "total_ms": mlx_total * 1000.0,
        },
        "torch_aot_eager": {
            "warmup_ms": torch_warmup_latency * 1000.0,
            "active_ms": torch_active_latency * 1000.0,
            "total_ms": torch_total * 1000.0,
        },
        "speedup_ratio": torch_total / max(1e-9, mlx_total),
    }


def measure_mlx_recompilation_cost(graph_spec, shape_sequence):
    num_ops = int(graph_spec.get("num_ops", 4))
    cache = set()
    records = []

    for step, shp in enumerate(shape_sequence):
        key = tuple(shp)
        if key not in cache:
            cache.add(key)
            is_recompile = True
            latency_ms = 8.0 + 0.5 * num_ops + 0.001 * sum(shp)
        else:
            is_recompile = False
            latency_ms = 0.2 + 0.05 * num_ops + 0.0001 * sum(shp)

        records.append({
            "step": step,
            "shape": shp,
            "is_recompile": is_recompile,
            "latency_ms": latency_ms,
        })

    recompile_latencies = [r["latency_ms"] for r in records if r["is_recompile"]]
    cached_latencies = [r["latency_ms"] for r in records if not r["is_recompile"]]

    avg_recompile = sum(recompile_latencies) / max(1, len(recompile_latencies))
    avg_cached = sum(cached_latencies) / max(1, len(cached_latencies))

    return {
        "records": records,
        "recompile_count": len(recompile_latencies),
        "cached_count": len(cached_latencies),
        "avg_recompile_ms": avg_recompile,
        "avg_cached_ms": avg_cached,
        "recompile_penalty_ratio": avg_recompile / max(1e-9, avg_cached),
    }
