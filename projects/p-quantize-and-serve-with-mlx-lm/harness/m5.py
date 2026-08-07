def check(workdir):
    import ref
    from mlx_serve.memory import check_stability
    allocs = [10.0, 10.0, 10.001, 10.002]
    res = check_stability(allocs)
    ref_res = ref.check_memory_stability(allocs)
    ok = 1.0 if res.get("memory_stable_ok") == ref_res["memory_stable_ok"] else 0.0
    return {"memory_stable_ok": ok}
