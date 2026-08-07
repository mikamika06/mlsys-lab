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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_redundant_prefill": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import disagg.p1d as p1d_mod
    orig_pipe = p1d_mod.Pipeline1P1D

    class BrokenPipeline(orig_pipe):
        def process_request(self, request_id, prompt_tokens, decode_steps):
            kv_payload = self.prefill_worker.process_prefill(request_id, prompt_tokens)
            curr_token = prompt_tokens[-1]
            tokens_out = []
            for _ in range(decode_steps):
                curr_token = self.decode_worker.step_decode(request_id, curr_token)
                tokens_out.append(curr_token)
            return {
                "request_id": request_id,
                "tokens": tokens_out,
                "prefill_stats": dict(self.prefill_worker.stats),
                "decode_stats": dict(self.decode_worker.stats)
            }

    p1d_mod.Pipeline1P1D = BrokenPipeline
    try:
        out["catches_redundant_prefill"] = 0.0 if _survives(path) else 1.0
    finally:
        p1d_mod.Pipeline1P1D = orig_pipe

    return out
