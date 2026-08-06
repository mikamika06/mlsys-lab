import ref


def check(workdir):
    from specdec.metrics import measure_acceptance_rate
    from specdec.optimizer import find_optimal_draft_max

    out = {"acceptance_rate_match": 0.0, "optimal_max_match": 0.0}

    traces = [(5, 3), (5, 5), (5, 1), (5, 2)]
    want_rate = ref.compute_acceptance(traces)
    try:
        got_rate = measure_acceptance_rate(traces)
    except Exception as e:
        out["_note"] = f"measure_acceptance_rate raised {e}"
        return out

    if isinstance(got_rate, (int, float)) and abs(float(got_rate) - float(want_rate)) < 1e-5:
        out["acceptance_rate_match"] = 1.0
    else:
        out["_note"] = f"measure_acceptance_rate got {got_rate}, want {want_rate}"
        return out

    max_g, p_val, cost = 6, 0.7, 0.25
    want_opt = ref.compute_optimal(max_g, p_val, cost)
    try:
        got_opt = find_optimal_draft_max(max_g, p_val, cost)
    except Exception as e:
        out["_note"] = f"find_optimal_draft_max raised {e}"
        return out

    if int(got_opt) == int(want_opt):
        out["optimal_max_match"] = 1.0
    else:
        out["_note"] = f"find_optimal_draft_max got {got_opt}, want {want_opt}"
        return out

    return out
