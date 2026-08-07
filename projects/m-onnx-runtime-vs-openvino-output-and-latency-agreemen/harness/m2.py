import ref


def check(workdir):
    from bench.xeon_ranking import check_precision_fairness, rank_xeon_throughput

    logs = ref.generate_xeon_logs()
    want_ranking = ref.build_xeon_ranking(logs)
    fairness_cases = ref.build_fairness_checks()

    out = {"rankings_matched": 0.0, "fairness_matched": 0.0}

    try:
        got_ranking = rank_xeon_throughput(logs)
        if len(got_ranking) == len(want_ranking):
            match = True
            for g, w in zip(got_ranking, want_ranking):
                if g["engine"] != w["engine"] or abs(g["median_qps"] - w["median_qps"]) > 1e-4:
                    match = False
                    break
            if match:
                out["rankings_matched"] = 1.0
            else:
                out["_note"] = f"Ranking mismatch. Got {got_ranking}, want {want_ranking}"
        else:
            out["_note"] = f"Ranking length mismatch. Got {len(got_ranking)}, want {len(want_ranking)}"
    except Exception as e:
        out["_note"] = f"rank_xeon_throughput failed: {type(e).__name__}: {str(e)[:120]}"
        return out

    try:
        fair_ok = True
        for case in fairness_cases:
            res = check_precision_fairness(case["engine_a"], case["engine_b"])
            if res.get("fair") != case["fair"]:
                fair_ok = False
                break
        if fair_ok:
            out["fairness_matched"] = 1.0
        elif "_note" not in out:
            out["_note"] = "Fairness check output mismatch"
    except Exception as e:
        if "_note" not in out:
            out["_note"] = f"check_precision_fairness failed: {type(e).__name__}: {str(e)[:120]}"

    return out
