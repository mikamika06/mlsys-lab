import ref
import sys

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    out = {"catches_stop_and_truncation": 0.0}

    if ref.run_tests(workdir) is not True:
        return out

    restore3 = ref.apply_regression(workdir, 3)
    try:
        c3 = not ref.survives(workdir)
    finally:
        restore3()

    restore4 = ref.apply_regression(workdir, 4)
    try:
        c4 = not ref.survives(workdir)
    finally:
        restore4()

    if c3 and c4:
        out["catches_stop_and_truncation"] = 1.0
    return out
