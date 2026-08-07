import ref


def check(workdir):
    from lanewaste.metrics import calculate_wasted_lane_time

    out = {"metrics_matched": 0.0, "total": float(len(ref.TEST_WORKLOADS))}
    matched = 0
    for i, item in enumerate(ref.TEST_WORKLOADS):
        n = item["n"]
        overhead = item["launch_overhead"]
        cands = item["candidates"]
        ok = True
        for b in cands:
            want = ref.calculate_wasted_lane_time(n, b, overhead)
            got = calculate_wasted_lane_time(n, b, overhead)
            if abs(want - got) > 1e-5:
                ok = False
                if "_note" not in out:
                    out["_note"] = (
                        f"workload {i} (n={n}, b={b}): got {got}, want {want}"
                    )
                break
        if ok:
            matched += 1
    out["metrics_matched"] = float(matched)
    return out
