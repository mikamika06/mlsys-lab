import sys

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import importlib
    engine_mod = importlib.import_module("runner.engine")
    bench_mod = importlib.import_module("runner.bench")
    profiler_mod = importlib.import_module("runner.profiler")

    out = {"timing_decomposed": 0.0, "percentages_valid": 0.0, "queue_detected": 0.0}

    try:
        cfg = engine_mod.EngineConfig()
        engine = engine_mod.Engine(cfg)
        bench = bench_mod.LoadBench(warmup_runs=0)
        wl = bench.generate_workload(num_users=4, prompt_len=32, output_len=50)
        res = bench.run_benchmark(engine, wl)
        metrics = res["metrics"]
        decomp = profiler_mod.decompose_timing(metrics)
    except Exception:
        return out

    if not isinstance(decomp, dict):
        return out

    required = {"avg_queue_ms", "avg_prefill_ms", "avg_decode_ms", "queue_pct", "prefill_pct", "decode_pct"}
    if required.issubset(decomp.keys()):
        out["timing_decomposed"] = 1.0

    pct_sum = decomp.get("queue_pct", 0) + decomp.get("prefill_pct", 0) + decomp.get("decode_pct", 0)
    if abs(pct_sum - 100.0) < 1e-2:
        out["percentages_valid"] = 1.0

    if decomp.get("avg_queue_ms", 0) > 0 and decomp.get("queue_pct", 0) > 0:
        out["queue_detected"] = 1.0

    return out
