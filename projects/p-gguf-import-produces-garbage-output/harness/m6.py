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
    out = {"has_tests": 0.0, "passes_on_good": 0.0,
           "catches_broken_importer": 0.0, "catches_broken_tokenizer": 0.0,
           "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import gguf_engine.importer as imp_mod
    import gguf_engine.tokenizer as tok_mod

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_verify = imp_mod.GGUFImporter.verify_metadata
    def bad_verify(self):
        return {}
    imp_mod.GGUFImporter.verify_metadata = bad_verify
    try:
        out["catches_broken_importer"] = 0.0 if _survives(path) else 1.0
    finally:
        imp_mod.GGUFImporter.verify_metadata = good_verify

    good_template = tok_mod.GGUFTokenizer.apply_chat_template
    def bad_template(self, messages):
        return ""
    tok_mod.GGUFTokenizer.apply_chat_template = bad_template
    try:
        out["catches_broken_tokenizer"] = 0.0 if _survives(path) else 1.0
    finally:
        tok_mod.GGUFTokenizer.apply_chat_template = good_template

    out["faults_caught"] = out["catches_broken_importer"] + out["catches_broken_tokenizer"]
    return out
