import ref
import sys

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    out = {"catches_tokenization": 0.0}

    if ref.run_tests(workdir) is not True:
        return out

    restore = ref.apply_regression(workdir, 2)
    try:
        if not ref.survives(workdir):
            out["catches_tokenization"] = 1.0
    finally:
        restore()
    return out
