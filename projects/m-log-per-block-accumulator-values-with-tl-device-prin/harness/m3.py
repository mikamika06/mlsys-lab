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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_regex": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
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

    import debugger.masking as m
    import debugger.logger as lg
    import re

    good_extract = m.extract_program_id
    good_parse = lg.parse_device_print

    # Inject a fault: regex that only matches single-digit coordinates
    def bad_extract(error_log: str) -> tuple:
        pattern = re.compile(r"program_id\s*\(\s*(\d)\s*,\s*(\d)\s*,\s*(\d)\s*\)")
        match = pattern.search(error_log)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return ()

    def bad_parse(stdout: str) -> dict:
        result = {}
        pattern = re.compile(r"\[(\d),\s*(\d),\s*(\d)\]\s*([^:]+):\s*([0-9.-]+)")
        for line in stdout.splitlines():
            match = pattern.search(line)
            if match:
                pid = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                result[pid] = float(match.group(5))
        return result

    m.extract_program_id = bad_extract
    lg.parse_device_print = bad_parse

    try:
        if not _survives(path):
            out["catches_bad_regex"] = 1.0
        else:
            out["_note"] = "tests passed even when multi-digit regex was broken"
    finally:
        m.extract_program_id = good_extract
        lg.parse_device_print = good_parse

    return out
