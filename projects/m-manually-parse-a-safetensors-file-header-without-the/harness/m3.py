import importlib.util
import os
import sys


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unmapped_keys": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"tests fail on good reference code: {type(e).__name__}: {str(e)[:120]}"
        )
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import safetensors_interop.remap as remap_mod

    orig_fn = remap_mod.remap_hf_to_mlx

    def broken_remap(hf_tensors, rule_map):
        return ({k: v for k, v in hf_tensors.items()}, [])

    remap_mod.remap_hf_to_mlx = broken_remap

    try:
        if _survives(path):
            out["catches_unmapped_keys"] = 0.0
            out["_note"] = (
                "test passed even when remap_hf_to_mlx returned untranslated raw HF keys"
            )
        else:
            out["catches_unmapped_keys"] = 1.0
    finally:
        remap_mod.remap_hf_to_mlx = orig_fn

    return out
