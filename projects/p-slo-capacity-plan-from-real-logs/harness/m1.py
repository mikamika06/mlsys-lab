import os
import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    import capacity.planner as planner

    m_path, b_path = ref.generate_fixtures(workdir)

    out = {"time_series_parsed": 0.0, "bench_parsed": 0.0}

    try:
        ts = planner.parse_metrics_series(m_path)
        ref_ts = ref.oracle_parse_metrics(m_path)
        if (
            len(ts["timestamp"]) == len(ref_ts["timestamp"])
            and abs(float(ts["gpu_utilization"].mean() - ref_ts["gpu_utilization"].mean())) < 1e-5
        ):
            out["time_series_parsed"] = 1.0
    except Exception:
        pass

    try:
        bench = planner.parse_bench_results(b_path)
        ref_bench = ref.oracle_parse_bench(b_path)
        if len(bench) == len(ref_bench) and bench[0]["offered_rps"] == ref_bench[0]["offered_rps"]:
            out["bench_parsed"] = 1.0
    except Exception:
        pass

    return out
