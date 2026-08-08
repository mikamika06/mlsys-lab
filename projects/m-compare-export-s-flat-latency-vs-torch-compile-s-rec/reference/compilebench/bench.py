def compare_latency(requests, config):
    """Compare latency profiles between torch.compile and torch.export."""
    compile_cost = config["compile_cost_ms"]
    base_exec = config["base_exec_ms"]
    per_batch = config["per_batch_ms"]
    export_overhead = config["export_overhead_ms"]
    max_batch = config["max_batch_size"]

    seen_shapes = set()
    compile_latencies = []
    export_latencies = []
    recompile_count = 0

    for req in requests:
        b = req["batch_size"]
        if b > max_batch:
            raise ValueError(f"Batch size {b} exceeds max allowed {max_batch}")

        exec_time = base_exec + b * per_batch

        if b not in seen_shapes:
            compile_lat = exec_time + compile_cost
            seen_shapes.add(b)
            recompile_count += 1
        else:
            compile_lat = exec_time
        compile_latencies.append(compile_lat)

        export_lat = exec_time + export_overhead
        export_latencies.append(export_lat)

    max_compile = max(compile_latencies) if compile_latencies else 0.0
    max_export = max(export_latencies) if export_latencies else 1.0
    sum_compile = sum(compile_latencies)
    sum_export = sum(export_latencies) if export_latencies else 1.0

    return {
        "compile_latencies": compile_latencies,
        "export_latencies": export_latencies,
        "recompile_count": recompile_count,
        "max_spike_ratio": max_compile / max_export if max_export > 0 else 0.0,
        "total_latency_ratio": sum_compile / sum_export if sum_export > 0 else 0.0
    }
