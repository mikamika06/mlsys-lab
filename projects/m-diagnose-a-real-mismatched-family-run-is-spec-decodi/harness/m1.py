import ref

def check(workdir):
    from specdiag.metrics import compute_acceptance_metrics
    out = {"metrics_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.compute_acceptance_metrics(cfg)
        got = compute_acceptance_metrics(cfg)
        if isinstance(got, dict) and abs(got.get("expected_accepted", -1) - want["expected_accepted"]) < 1e-5:
            ok += 1
    out["metrics_matched"] = float(ok)
    return out
