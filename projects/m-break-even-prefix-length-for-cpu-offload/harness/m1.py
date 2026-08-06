import math
import ref


def _close(a, b, tol=1e-4):
    if math.isinf(a) and math.isinf(b):
        return True
    return abs(a - b) / (abs(b) + 1e-9) < tol


def check(workdir):
    from offload.simulator import (
        compute_breakeven_prefix_length,
        compute_kv_bytes,
        estimate_load_time,
        estimate_recompute_time,
    )

    out = {"simulations_matched": 0.0, "breakevens_matched": 0.0}
    sim_ok = 0
    be_ok = 0

    for i in range(len(ref.CONFIGS)):
        cfg = ref.CONFIGS[i]
        hw = ref.HARDWARE[i]
        tier = ref.TIERS[i]["ram"]
        prefix_len = (i + 1) * 1024

        want_kv = ref.compute_kv_bytes(cfg, prefix_len)
        got_kv = compute_kv_bytes(cfg, prefix_len)

        want_recomp = ref.estimate_recompute_time(cfg, hw, prefix_len)
        got_recomp = estimate_recompute_time(cfg, hw, prefix_len)

        want_load = ref.estimate_load_time(cfg, tier, prefix_len)
        got_load = estimate_load_time(cfg, tier, prefix_len)

        if got_kv == want_kv and _close(got_recomp, want_recomp) and _close(got_load, want_load):
            sim_ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: kv={got_kv} vs {want_kv}, recomp={got_recomp} vs {want_recomp}, load={got_load} vs {want_load}"

        want_be = ref.compute_breakeven_prefix_length(cfg, hw, tier)
        got_be = compute_breakeven_prefix_length(cfg, hw, tier)

        if _close(got_be, want_be):
            be_ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: breakeven got {got_be}, ref {want_be}"

    out["simulations_matched"] = float(sim_ok)
    out["breakevens_matched"] = float(be_ok)
    return out
