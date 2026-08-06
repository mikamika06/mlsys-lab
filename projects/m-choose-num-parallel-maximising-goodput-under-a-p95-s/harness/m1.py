import ref


def check(workdir):
    from runner.capacity import select_optimal_num_parallel

    benchmarks = ref.generate_benchmark_data()
    p95_slo_ms = 200.0

    res = select_optimal_num_parallel(benchmarks, p95_slo_ms)
    oracle = ref.select_optimal_num_parallel_oracle(benchmarks, p95_slo_ms)

    out = {
        "optimal_concurrency_matched": 0.0,
        "max_goodput_matched": 0.0,
    }

    if res and res.get("num_parallel") == oracle["num_parallel"]:
        out["optimal_concurrency_matched"] = 1.0

    if res and abs(res.get("max_goodput", 0.0) - oracle["max_goodput"]) < 1e-3:
        out["max_goodput_matched"] = 1.0
    else:
        out["_note"] = f"Expected max_goodput {oracle['max_goodput']}, got {res.get('max_goodput') if res else 'None'}"

    return out
