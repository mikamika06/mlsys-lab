import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    import capacity.planner as planner

    _, b_path = ref.generate_fixtures(workdir)
    bench_data = ref.oracle_parse_bench(b_path)

    out = {"knee_capacity_found": 0.0}
    try:
        knee = planner.find_knee_capacity(bench_data, target_p95=2.5)
        if abs(knee - 30.0) < 1e-3:
            out["knee_capacity_found"] = 1.0
    except Exception:
        pass

    return out
