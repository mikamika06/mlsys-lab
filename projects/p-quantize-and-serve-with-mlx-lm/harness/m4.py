def check(workdir):
    import ref
    from mlx_serve.load import benchmark_load
    latencies = [0.1, 0.2, 0.3, 0.4, 0.5, 1.5]
    res = benchmark_load(concurrency=3, latencies=latencies)
    ref_res = ref.run_load_test(3, latencies)
    ok = 1.0 if res.get("load_p95_ok") == ref_res["load_p95_ok"] else 0.0
    return {"load_p95_ok": ok}
