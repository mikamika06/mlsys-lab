import sys

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import importlib
    engine_mod = importlib.import_module("runner.engine")
    bench_mod = importlib.import_module("runner.bench")
    profiler_mod = importlib.import_module("runner.profiler")

    out = {"knee_shifted": 0.0, "p95_improved": 0.0}

    try:
        default_cfg = engine_mod.EngineConfig()
        opt_cfg = profiler_mod.optimize_config(default_cfg)

        bench = bench_mod.LoadBench(warmup_runs=0)
        wl4 = bench.generate_workload(num_users=4, prompt_len=32, output_len=50)

        e_def = engine_mod.Engine(default_cfg)
        res_def = bench.run_benchmark(e_def, wl4)
        p95_def = res_def["p95_latency_ms"]

        e_opt = engine_mod.Engine(opt_cfg)
        res_opt = bench.run_benchmark(e_opt, wl4)
        p95_opt = res_opt["p95_latency_ms"]

        slot_counts = [1, 2, 3, 4, 5]
        opt_curve = profiler_mod.build_slot_scaling_curve(opt_cfg, slot_counts)
        opt_knee = profiler_mod.find_knee(opt_curve)

        if opt_knee >= 4:
            out["knee_shifted"] = 1.0

        if p95_opt < p95_def * 0.7:
            out["p95_improved"] = 1.0
    except Exception:
        pass

    return out
