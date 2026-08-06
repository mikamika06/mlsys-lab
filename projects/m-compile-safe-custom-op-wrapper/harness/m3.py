import importlib.util
import os
import torch

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod)
           if n.startswith("test_") and callable(getattr(mod, n))]
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_graph_break": 0.0}
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

    import custom_op.wrapper as w
    orig_run = w.run_attention

    def broken_run_attention(q, k, v):
        torch._dynamo.graph_break()
        scale = 1.0 / (q.shape[-1] ** 0.5)
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

    w.run_attention = broken_run_attention
    try:
        out["catches_graph_break"] = 0.0 if _survives(path) else 1.0
    finally:
        w.run_attention = orig_run

    return out
