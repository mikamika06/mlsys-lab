import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    import capacity.planner as planner

    _, b_path = ref.generate_fixtures(workdir)
    bench_data = ref.oracle_parse_bench(b_path)

    out = {"goodput_correct": 0.0}
    try:
        res = planner.calculate_goodput(bench_data, max_p95_latency=2.5)
        if abs(res["goodput_rps"] - 28.5) < 1e-3 and res["valid_runs"] == 3:
            out["goodput_correct"] = 1.0
    except Exception:
        pass

    return out
