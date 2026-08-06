import importlib.util
import os
import sys

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.dirname(path)))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_overwritten_backup": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
        
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
        
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
        
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    sys.path.insert(0, workdir)
    try:
        import gguf_chat.modifier as mod
        good = mod.set_chat_template

        def buggy_set_chat_template(reader, writer, template_str):
            current_field = reader.fields.get("tokenizer.chat_template")
            if current_field is not None:
                current_str = bytes(current_field.parts[-1]).decode("utf-8")
                writer.add_string("tokenizer.chat_template.backup", current_str)
            writer.add_string("tokenizer.chat_template", template_str)

        mod.set_chat_template = buggy_set_chat_template
        try:
            out["catches_overwritten_backup"] = 0.0 if _survives(path) else 1.0
        finally:
            mod.set_chat_template = good
    finally:
        sys.path.pop(0)

    return out
