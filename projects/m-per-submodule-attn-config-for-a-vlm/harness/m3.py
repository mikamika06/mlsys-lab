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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_merged_submodules": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct plan: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import vlmcfg.parser as p
    good = p.build_configs

    def broken(config):
        buckets = {}
        for sub in config["submodules"]:
            key = (sub["kind"], sub["kv_heads"], sub["head_dim"])
            buckets.setdefault(key, []).append(sub)
        out_ = []
        for (kind, kv, hd), rows in sorted(buckets.items()):
            out_.append({
                "kind": kind,
                "num_heads": max(r["num_heads"] for r in rows),
                "kv_heads": kv,
                "head_dim": hd,
                "causal": all(r["causal"] for r in rows),
                "submodules": sorted(r["index"] for r in rows)
            })
        return out_

    p.build_configs = broken
    import vlmcfg
    if hasattr(vlmcfg, "build_configs"):
        vlmcfg.build_configs = broken
    try:
        out["catches_merged_submodules"] = 0.0 if _survives(path) else 1.0
    finally:
        p.build_configs = good
        if hasattr(vlmcfg, "build_configs"):
            vlmcfg.build_configs = good
    return out
