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

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_axis": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {e}"
        return out

    if first is None:
        out["_note"] = "no test_ functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import inference
    good_run = inference.run_hand_rolled

    def bad_axis_run(model, prompt_ids, max_tokens):
        import numpy as np
        prompt_ids = list(prompt_ids)
        generated = []
        for _ in range(max_tokens):
            logits = model.infer(prompt_ids)
            next_token = int(np.argmax(logits[0, 0, :]))
            generated.append(next_token)
            prompt_ids.append(next_token)
        return generated

    inference.run_hand_rolled = bad_axis_run
    try:
        survives = False
        try:
            survives = (_run(path) is True)
        except Exception:
            pass
        if not survives:
            out["catches_bad_axis"] = 1.0
    finally:
        inference.run_hand_rolled = good_run

    return out
