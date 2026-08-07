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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_inverted_dominance": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import streammetrics.classify as cl
    good = cl.classify_workload_dominance

    def inverted_classify(ttft, prefill_tok_per_sec, decode_tok_per_sec, prompt_tokens, completion_tokens):
        prefill_time = ttft
        decode_time = (completion_tokens - 1) / decode_tok_per_sec if decode_tok_per_sec > 0 else 0.0
        if prefill_time < decode_time:
            return "prefill-dominated"
        return "decode-dominated"

    cl.classify_workload_dominance = inverted_classify
    import streammetrics
    streammetrics.classify.classify_workload_dominance = inverted_classify

    try:
        out["catches_inverted_dominance"] = 0.0 if _survives(path) else 1.0
    finally:
        cl.classify_workload_dominance = good
        streammetrics.classify.classify_workload_dominance = good

    return out
