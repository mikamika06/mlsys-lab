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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_clip": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import opsetfix.transformer as t
    good_fn = t.fix_opset_and_clip

    def broken_fn(model_path, output_path, target_opset=13):
        import onnx
        model = onnx.load(model_path)
        for opset in model.opset_import:
            if opset.domain == "" or opset.domain == "ai.onnx":
                opset.version = target_opset
        onnx.save(model, output_path)
        return output_path

    t.fix_opset_and_clip = broken_fn
    try:
        out["catches_broken_clip"] = 0.0 if _survives(path) else 1.0
    finally:
        t.fix_opset_and_clip = good_fn
    return out
