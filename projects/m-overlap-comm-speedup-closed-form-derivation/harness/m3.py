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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_batch_formula": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on valid implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import dsengine.batch_config as bc
    good_validate = bc.validate_batch_config
    good_resolve = bc.resolve_batch_config

    def bad_validate(config):
        tbs = config.get("train_batch_size")
        mbs = config.get("train_micro_batch_size_per_gpu")
        gas = config.get("gradient_accumulation_steps")
        if tbs is None or mbs is None or gas is None:
            return False
        return tbs == mbs * gas

    def bad_resolve(config):
        res = dict(config)
        res["train_batch_size"] = res.get("train_micro_batch_size_per_gpu", 1) * res.get("gradient_accumulation_steps", 1)
        res.setdefault("data_parallel_size", 1)
        return res

    bc.validate_batch_config = bad_validate
    bc.resolve_batch_config = bad_resolve

    try:
        out["catches_invalid_batch_formula"] = 0.0 if _survives(path) else 1.0
    finally:
        bc.validate_batch_config = good_validate
        bc.resolve_batch_config = good_resolve

    return out
