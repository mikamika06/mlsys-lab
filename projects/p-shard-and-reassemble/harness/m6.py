import importlib.util
import os

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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_missing_shard": 0.0}
    if not os.path.isfile(path):
        return out

    try:
        import gguf_shard.sharder as shd
    except ImportError:
        return out

    try:
        first = _run(path)
    except Exception:
        out["has_tests"] = 1.0
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_reassemble = shd.reassemble

    def broken_reassemble(shards):
        from gguf_shard.model import Model
        meta = shards[0].metadata.copy()
        meta.pop("split.no", None)
        meta.pop("split.count", None)
        meta.pop("split.checksum", None)
        tensors = {}
        for s in shards:
            tensors.update(s.tensors)
        return Model(meta, tensors)

    shd.reassemble = broken_reassemble
    try:
        out["catches_missing_shard"] = 0.0 if _survives(path) else 1.0
    finally:
        shd.reassemble = good_reassemble

    return out
