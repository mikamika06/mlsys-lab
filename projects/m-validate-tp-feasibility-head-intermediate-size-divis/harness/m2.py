import math
import ref


def check(workdir):
    from tpval.traffic import compute_pp_bubble_fraction, compute_tp_traffic

    out = {"traffic_matches": 0.0, "bubble_rel_err": 1.0}

    traffic_ok = True
    rates = [100.0, 500.0, 2500.0]

    for cfg in ref.CONFIGS:
        for tp in ref.TP_DEGREES:
            for rate in rates:
                want = ref.compute_tp_traffic(cfg, tp, rate)
                got = compute_tp_traffic(cfg, tp, rate)
                b_diff = abs(got.get("bytes_per_token_per_rank", -1) - want["bytes_per_token_per_rank"])
                t_diff = abs(got.get("total_bus_bytes_per_sec", -1) - want["total_bus_bytes_per_sec"])
                if b_diff > 1e-4 or t_diff > 1e-4:
                    traffic_ok = False
                    out["_note"] = f"Traffic mismatch for TP={tp}, rate={rate}"
                    break
            if not traffic_ok:
                break
        if not traffic_ok:
            break

    if traffic_ok:
        out["traffic_matches"] = 1.0

    max_rel_err = 0.0
    test_cases = [
        (4, 2),
        (8, 4),
        (16, 4),
        (32, 8),
        (64, 8)
    ]

    for m, p in test_cases:
        want_b = ref.compute_pp_bubble_fraction(m, p)
        got_b = compute_pp_bubble_fraction(m, p)
        err = abs(got_b - want_b) / max(1e-9, abs(want_b))
        if err > max_rel_err:
            max_rel_err = err

    out["bubble_rel_err"] = float(max_rel_err)
    return out
