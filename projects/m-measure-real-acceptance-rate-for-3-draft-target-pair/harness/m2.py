import ref


def check(workdir):
    import sys

    sys.path.insert(0, workdir)
    from specpair.selector import select_optimal_draft

    out = {"optimal_draft_matched": 0.0, "throughput_ratio": 0.0}

    ref_res = ref.select_optimal_draft(
        ref.SELECTION_CANDIDATES, ref.TARGET_LATENCY, ref.GAMMA
    )
    try:
        got_res = select_optimal_draft(
            ref.SELECTION_CANDIDATES, ref.TARGET_LATENCY, ref.GAMMA
        )
    except Exception as e:
        out["_note"] = f"select_optimal_draft raised {type(e).__name__}: {str(e)}"
        return out

    if got_res.get("best_draft") == ref_res["best_draft"]:
        out["optimal_draft_matched"] = 1.0

    ref_tp = ref_res["throughput"]
    got_tp = got_res.get("throughput", 0.0)

    if ref_tp > 0 and got_tp > 0:
        out["throughput_ratio"] = got_tp / ref_tp

    return out
