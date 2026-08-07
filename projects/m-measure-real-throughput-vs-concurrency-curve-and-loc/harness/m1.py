import sys

import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"curve_matched": 0.0}
    try:
        from throughput.bench import measure_concurrency_curve

        ref_results = ref.measure_concurrency_curve(
            ref.CONCURRENCY_LEVELS, ref.NUM_REQUESTS, ref.WORKLOAD_SPEC
        )
        got_results = measure_concurrency_curve(
            ref.CONCURRENCY_LEVELS, ref.NUM_REQUESTS, ref.WORKLOAD_SPEC
        )

        if len(ref_results) != len(got_results):
            out["_note"] = f"Length mismatch: got {len(got_results)}, expected {len(ref_results)}"
            return out

        matched = True
        for r_item, g_item in zip(ref_results, got_results):
            for key in ["concurrency", "throughput_tps", "total_time_sec", "avg_latency_sec", "block_utilization"]:
                if abs(r_item[key] - g_item.get(key, -1.0)) > 1e-3:
                    matched = False
                    out["_note"] = f"Mismatch on key '{key}': got {g_item.get(key)}, expected {r_item[key]}"
                    break
            if not matched:
                break

        if matched:
            out["curve_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"Error running milestone 1 check: {type(e).__name__}: {str(e)[:120]}"

    return out
