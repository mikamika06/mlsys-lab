import ref

def check(workdir):
    from runners.compare import run_comparison
    from runners.metrics import compute_metrics
    out = {"comparison_match": 0.0, "rel_err_bound": 0.0}
    try:
        runners = [
            {"name": "llamafile", "scale": 1.0},
            {"name": "runner_b", "scale": 1.02},
            {"name": "runner_c", "scale": 0.98}
        ]
        workload = ref.WORKLOADS[0]
        want_comp = ref.run_comparison(runners, workload)
        got_comp = run_comparison(runners, workload)
        if got_comp == want_comp:
            out["comparison_match"] = 1.0

        want_metrics = ref.compute_metrics({"tokens": workload}, want_comp)
        got_metrics = compute_metrics({"tokens": workload}, got_comp)

        valid = True
        for wm, gm in zip(want_metrics, got_metrics):
            if abs(wm["max_rel_err"] - gm["max_rel_err"]) > 1e-5:
                valid = False
        if valid and len(got_metrics) == 3:
            out["rel_err_bound"] = 1.0
    except Exception as e:
        out["_note"] = f"error in m2 check: {str(e)[:120]}"
    return out
