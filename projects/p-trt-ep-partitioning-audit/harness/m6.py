import importlib.util
import os
import sys


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
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_disabled_cache": 0.0,
        "catches_broken_rewriter": 0.0,
        "faults_caught": 0.0
    }

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import trtep.cache as cache
    import trtep.rewriter as rewriter

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"learner tests fail on correct implementation: {type(e).__name__}: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    orig_build_or_load = cache.EngineCache.build_or_load

    def broken_build_or_load(self, subgraph, builder_fn):
        self.misses += 1
        return builder_fn(subgraph), False

    cache.EngineCache.build_or_load = broken_build_or_load
    try:
        out["catches_disabled_cache"] = 0.0 if _survives(path) else 1.0
    finally:
        cache.EngineCache.build_or_load = orig_build_or_load

    orig_rewrite = rewriter.rewrite_graph

    def nop_rewrite(graph, supported_ops):
        return graph

    rewriter.rewrite_graph = nop_rewrite
    try:
        out["catches_broken_rewriter"] = 0.0 if _survives(path) else 1.0
    finally:
        rewriter.rewrite_graph = orig_rewrite

    out["faults_caught"] = out["catches_disabled_cache"] + out["catches_broken_rewriter"]
    return out
