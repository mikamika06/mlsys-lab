import importlib.util
import os
import sys

sys.path.insert(0, ".")
import ref


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
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
    sys.path.insert(0, workdir)
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_global_gate_bug": 0.0,
        "catches_ppl_kl_swap_bug": 0.0,
    }

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out[
            "_note"
        ] = f"The tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import evalrec.gate as g
    import evalrec.metrics as m

    good_gate = g.evaluate_acceptance_gate
    good_rank = m.rank_quant_candidates

    def broken_global_gate(candidate_metrics, category_thresholds):
        mean_kl = sum(v["kl"] for v in candidate_metrics.values()) / max(
            len(candidate_metrics), 1
        )
        mean_ppl = sum(v["ppl"] for v in candidate_metrics.values()) / max(
            len(candidate_metrics), 1
        )
        max_kl = max(t["max_kl"] for t in category_thresholds.values())
        max_ppl = max(t["max_ppl"] for t in category_thresholds.values())
        accepted = mean_kl <= max_kl and mean_ppl <= max_ppl
        cat_results = {
            cat: {
                "kl_pass": accepted,
                "ppl_pass": accepted,
                "passed": accepted,
            }
            for cat in category_thresholds
        }
        return {
            "accepted": accepted,
            "category_results": cat_results,
            "failed_categories": (
                [] if accepted else sorted(category_thresholds.keys())
            ),
        }

    def broken_ppl_kl_swap(teacher_data, candidates_data):
        res = ref.rank_quant_candidates(teacher_data, candidates_data)
        for c, stats in res["candidates"].items():
            stats["kl_rank"], stats["ppl_rank"] = (
                stats["ppl_rank"],
                stats["kl_rank"],
            )
        return res

    try:
        g.evaluate_acceptance_gate = broken_global_gate
        out["catches_global_gate_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        g.evaluate_acceptance_gate = good_gate

    try:
        m.rank_quant_candidates = broken_ppl_kl_swap
        out["catches_ppl_kl_swap_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        m.rank_quant_candidates = good_rank

    return out
