import sys

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import importlib
    engine_mod = importlib.import_module("runner.engine")
    bench_mod = importlib.import_module("runner.bench")

    out = {"warmup_ok": 0.0, "metrics_ok": 0.0, "steady_state_clean": 0.0}

    try:
        cfg = engine_mod.EngineConfig()
        engine = engine_mod.Engine(cfg)
        bench = bench_mod.LoadBench(warmup_runs=2)
        wl = bench.generate_workload(num_users=2, prompt_len=32, output_len=50)
        res = bench.run_benchmark(engine, wl)
    except Exception:
        return out

    if not isinstance(res, dict):
        return out

    required_keys = {"warmup_completed", "num_users", "total_tokens", "total_wall_time_ms", "aggregate_tok_per_sec", "p95_latency_ms", "metrics"}
    if required_keys.issubset(res.keys()):
        out["metrics_ok"] = 1.0

    if res.get("warmup_completed") is True or res.get("warmup_completed") == 1:
        out["warmup_ok"] = 1.0

    if res.get("total_tokens") == 100 and len(res.get("metrics", [])) == 2:
        out["steady_state_clean"] = 1.0

    return out
