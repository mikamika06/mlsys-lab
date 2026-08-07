import ref


def check(workdir):
    from overlap.metrics import compute_metrics

    out = {"theoretical_min_rel_err": 1.0, "overlap_score_rel_err": 1.0}
    max_min_err = 0.0
    max_score_err = 0.0

    for events in ref.TRACES:
        want = ref.compute_metrics(events)
        got = compute_metrics(events)

        w_min = want["theoretical_min_step_time"]
        g_min = got.get("theoretical_min_step_time", 0.0)
        err_min = abs(g_min - w_min) / (abs(w_min) if abs(w_min) > 1e-9 else 1.0)
        if err_min > max_min_err:
            max_min_err = err_min

        w_score = want["overlap_efficiency_score"]
        g_score = got.get("overlap_efficiency_score", 0.0)
        err_score = abs(g_score - w_score) / (abs(w_score) if abs(w_score) > 1e-9 else 1.0)
        if err_score > max_score_err:
            max_score_err = err_score

    out["theoretical_min_rel_err"] = float(max_min_err)
    out["overlap_score_rel_err"] = float(max_score_err)
    return out
