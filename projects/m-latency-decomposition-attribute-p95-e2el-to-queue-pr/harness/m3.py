import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_throughput_ranking": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import latmetrics.goodput as g
    good_rank = g.rank_configs

    def faulty_rank_configs(configs, slo_ttft_ms, slo_tpot_ms):
        evaluated = []
        for cfg in configs:
            cid = cfg["config_id"]
            dur = cfg["duration_s"]
            reqs = cfg["requests"]
            res = g.evaluate_slo(reqs, slo_ttft_ms, slo_tpot_ms, dur)
            evaluated.append({
                "config_id": cid,
                "goodput": res["goodput"],
                "throughput": res["throughput"],
                "goodput_ratio": res["goodput_ratio"],
            })
        evaluated.sort(key=lambda x: (-x["throughput"], -x["goodput"], str(x["config_id"])))
        for rank, item in enumerate(evaluated, start=1):
            item["rank"] = rank
        return evaluated

    g.rank_configs = faulty_rank_configs
    import latmetrics
    latmetrics.goodput.rank_configs = faulty_rank_configs

    try:
        out["catches_throughput_ranking"] = 0.0 if _survives(path) else 1.0
    finally:
        g.rank_configs = good_rank
        latmetrics.goodput.rank_configs = good_rank

    return out
