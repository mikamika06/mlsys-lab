import importlib.util
import os
import sys
import ref


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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {
        "report_sorted_ok": 0.0,
        "report_metrics_ok": 0.0,
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_broken_classify": 0.0,
        "catches_unsorted_report": 0.0,
        "faults_caught": 0.0,
    }

    sys.path.insert(0, workdir)
    import roofline.analysis as analysis_mod
    import roofline.model as model_mod

    hw = ref.get_hw_spec()
    raw = ref.generate_raw_profile(47)
    agg = ref.oracle_aggregate_profile(raw)

    try:
        learner_rep = analysis_mod.generate_prioritized_report(agg, hw)
        oracle_rep = ref.oracle_generate_prioritized_report(agg, hw)

        is_sorted = all(
            learner_rep[i]["potential_savings_us"] >= learner_rep[i + 1]["potential_savings_us"]
            for i in range(len(learner_rep) - 1)
        )
        if is_sorted and len(learner_rep) == len(oracle_rep):
            out["report_sorted_ok"] = 1.0

        metrics_match = True
        for lr, oracle_r in zip(learner_rep, oracle_rep):
            if lr["name"] != oracle_r["name"]:
                metrics_match = False
                break
            if abs(lr["potential_savings_us"] - oracle_r["potential_savings_us"]) > 1e-3:
                metrics_match = False
                break
            if abs(lr["time_share_pct"] - oracle_r["time_share_pct"]) > 1e-3:
                metrics_match = False
                break
        if metrics_match:
            out["report_metrics_ok"] = 1.0
    except Exception:
        pass

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"learner tests fail on good implementation: {type(e).__name__}: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    orig_classify = model_mod.classify_kernel
    model_mod.classify_kernel = lambda intensity, hw: "broken_class"
    try:
        out["catches_broken_classify"] = 0.0 if _survives(path) else 1.0
    finally:
        model_mod.classify_kernel = orig_classify

    orig_gen_rep = analysis_mod.generate_prioritized_report

    def bad_report(aggregated, hw_spec):
        rep = orig_gen_rep(aggregated, hw_spec)
        rep.reverse()
        return rep

    analysis_mod.generate_prioritized_report = bad_report
    try:
        out["catches_unsorted_report"] = 0.0 if _survives(path) else 1.0
    finally:
        analysis_mod.generate_prioritized_report = orig_gen_rep

    out["faults_caught"] = out["catches_broken_classify"] + out["catches_unsorted_report"]
    return out
