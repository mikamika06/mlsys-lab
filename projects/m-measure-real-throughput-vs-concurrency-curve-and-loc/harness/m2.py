import sys

import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"knee_matched": 0.0, "throughput_ratio": 0.0}
    try:
        from throughput.analyzer import evaluate_concurrency_capacity, locate_knee

        ref_curve = ref.measure_concurrency_curve(
            ref.CONCURRENCY_LEVELS, ref.NUM_REQUESTS, ref.WORKLOAD_SPEC
        )
        c_levels = [r["concurrency"] for r in ref_curve]
        tps_list = [r["throughput_tps"] for r in ref_curve]

        want_knee = ref.locate_knee(c_levels, tps_list)
        got_knee = locate_knee(c_levels, tps_list)

        if got_knee == want_knee:
            out["knee_matched"] = 1.0

        ratio = evaluate_concurrency_capacity(c_levels, tps_list, got_knee)
        out["throughput_ratio"] = float(round(ratio, 4))
    except Exception as e:
        out["_note"] = f"Error running milestone 2 check: {type(e).__name__}: {str(e)[:120]}"

    return out
