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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_merged_shards": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import sharder.plan as p
    good_plan = p.build_shard_plan

    def broken_plan(tensors, max_bytes):
        names = [t["name"] for t in tensors]
        sz = 0
        for t in tensors:
            s = 4
            for d in t["shape"]:
                s *= d
            sz += s
        return [{"tensors": names, "size": sz}]

    p.build_shard_plan = broken_plan
    import sharder.schedule as s
    old_sched = s.compute_conversion_schedule

    def broken_schedule(vocab, tensors, max_bytes):
        from sharder.vocab import export_vocab_only
        return {"vocab": export_vocab_only(vocab), "shards": broken_plan(tensors, max_bytes)}

    s.compute_conversion_schedule = broken_schedule
    try:
        out["catches_merged_shards"] = 0.0 if _survives(path) else 1.0
    finally:
        p.build_shard_plan = good_plan
        s.compute_conversion_schedule = old_sched
    return out
