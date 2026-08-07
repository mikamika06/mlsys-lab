import ref


def check(workdir):
    from tokensweep.sweep import run_sweep
    workload = ref.generate_workload(123)
    budgets = [512, 1024, 2048, 4096, 8192]
    want = ref.run_sweep(workload, budgets)
    try:
        got = run_sweep(workload, budgets)
    except Exception as e:
        return {"sweep_matched": 0.0, "_note": f"run_sweep raised {type(e).__name__}: {e}"}
    if not isinstance(got, list) or len(got) != len(want):
        return {"sweep_matched": 0.0, "_note": f"expected list length {len(want)}, got {type(got)}"}
    match = 0
    for g, w in zip(got, want):
        if abs(g.get("ttft", 0) - w["ttft"]) < 1e-5 and abs(g.get("itl", 0) - w["itl"]) < 1e-5:
            match += 1
    out = {"sweep_matched": 1.0 if match == len(want) else 0.0}
    if match != len(want):
        out["_note"] = f"matched {match}/{len(want)} sweep points"
    return out
