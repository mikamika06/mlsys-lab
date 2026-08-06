import math
import ref


def _close_list(a, b, tol=1e-4):
    if len(a) != len(b):
        return False
    return all(abs(x - y) / (abs(y) + 1e-9) < tol for x, y in zip(a, b))


def check(workdir):
    from offload.tiers import evaluate_workload_latencies, select_optimal_tier

    out = {"tiers_matched": 0.0, "workloads_matched": 0.0}
    tier_ok = 0
    workload_ok = 0

    for i in range(len(ref.CONFIGS)):
        cfg = ref.CONFIGS[i]
        hw = ref.HARDWARE[i]
        tiers = ref.TIERS[i]
        prefix_len = 512 * (i + 1)
        max_lat = 0.01 * (i + 1)

        want_tier = ref.select_optimal_tier(cfg, hw, tiers, prefix_len, max_lat)
        got_tier = select_optimal_tier(cfg, hw, tiers, prefix_len, max_lat)

        if got_tier == want_tier:
            tier_ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: tier got {got_tier}, ref {want_tier}"

        workload = [256, 1024, 4096, 16384]
        want_eval = ref.evaluate_workload_latencies(cfg, hw, tiers, workload)
        got_eval = evaluate_workload_latencies(cfg, hw, tiers, workload)

        match = True
        if set(got_eval.keys()) != set(want_eval.keys()):
            match = False
        else:
            for k in want_eval:
                if not _close_list(got_eval[k], want_eval[k]):
                    match = False
                    break

        if match:
            workload_ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: workload mismatch for keys {list(got_eval.keys())}"

    out["tiers_matched"] = float(tier_ok)
    out["workloads_matched"] = float(workload_ok)
    return out
