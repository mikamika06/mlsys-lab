import ref


def check(workdir):
    from eagle.server import run_server_simulation
    from eagle.metrics import compute_eagle_metrics

    out = {"latency_ratio_ok": 0.0, "acceptance_rate_valid": 0.0}
    ok_lr = 0
    ok_ar = 0
    for r in ref.RUNS:
        sim = run_server_simulation(r)
        got = compute_eagle_metrics(r, sim)
        want = ref.compute_metrics(r, ref.simulate_server(r))
        if abs(got.get("acceptance_rate", 0) - want["acceptance_rate"]) < 1e-5:
            ok_ar += 1
        if abs(got.get("latency_ratio", 0) - want["latency_ratio"]) < 1e-5:
            ok_lr += 1
    if ok_ar == len(ref.RUNS):
        out["acceptance_rate_valid"] = 1.0
    if ok_lr == len(ref.RUNS):
        out["latency_ratio_ok"] = 1.0
    return out
