import ref
import sys

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "catches_determinism": 0.0}

    res = ref.run_tests(workdir)
    if res is None:
        return out
    out["has_tests"] = 1.0

    if res is not True:
        return out

    restore = ref.apply_regression(workdir, 1)
    try:
        if not ref.survives(workdir):
            out["catches_determinism"] = 1.0
    finally:
        restore()
    return out
