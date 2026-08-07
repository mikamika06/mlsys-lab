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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_tail_ignoring_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct profiler: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import sla.profiler as prof

    good_classify = prof.classify_sla_compliance

    def buggy_classify_only_median(batch_profiles, target_sla):
        results = {}
        max_compliant = None
        sorted_batches = sorted(batch_profiles.keys())
        target_pcts = sorted(list(target_sla.keys()))

        for b in sorted_batches:
            lats = batch_profiles[b]["latencies"]
            pcts = prof.calculate_percentiles(lats, target_pcts)
            violations = []
            if 50.0 in target_sla and pcts.get(50.0, 0) > target_sla[50.0]:
                violations.append(50.0)

            is_compliant = len(violations) == 0
            results[b] = {
                "compliant": is_compliant,
                "percentiles": pcts,
                "violations": violations,
            }
            if is_compliant:
                max_compliant = b

        return {
            "results": results,
            "max_compliant_batch": max_compliant,
        }

    prof.classify_sla_compliance = buggy_classify_only_median
    import sla
    sla.classify_sla_compliance = buggy_classify_only_median

    try:
        out["catches_tail_ignoring_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        prof.classify_sla_compliance = good_classify
        sla.classify_sla_compliance = good_classify

    return out
