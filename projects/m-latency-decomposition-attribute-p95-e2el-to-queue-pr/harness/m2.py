import ref


def check(workdir):
    from latmetrics.goodput import evaluate_slo, rank_configs

    out = {"ranking_matched": 0.0, "goodput_computed": 0.0}

    slo_ok = True
    for cfg in ref.CONFIGS:
        want_e = ref.evaluate_slo(cfg["requests"], ref.SLO_TTFT_MS, ref.SLO_TPOT_MS, cfg["duration_s"])
        got_e = evaluate_slo(cfg["requests"], ref.SLO_TTFT_MS, ref.SLO_TPOT_MS, cfg["duration_s"])
        for k, want_v in want_e.items():
            got_v = got_e.get(k, 0.0)
            if abs(got_v - want_v) > 1e-5:
                slo_ok = False
                out["_note"] = f"evaluate_slo mismatch on key {k}: got {got_v}, want {want_v}"
                break
        if not slo_ok:
            break

    if slo_ok:
        out["goodput_computed"] = 1.0

    want_rank = ref.rank_configs(ref.CONFIGS, ref.SLO_TTFT_MS, ref.SLO_TPOT_MS)
    got_rank = rank_configs(ref.CONFIGS, ref.SLO_TTFT_MS, ref.SLO_TPOT_MS)

    want_ids = [r["config_id"] for r in want_rank]
    got_ids = [r.get("config_id") for r in (got_rank or [])]

    if want_ids == got_ids:
        out["ranking_matched"] = 1.0
    elif "_note" not in out:
        out["_note"] = f"ranking mismatch: got {got_ids[:3]}, want {want_ids[:3]}"

    return out
