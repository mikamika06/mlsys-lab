import ref
import sys

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    out = {"caught_count": 0.0}

    if ref.run_tests(workdir) is not True:
        return out

    caught = 0
    for i in range(1, 6):
        restore = ref.apply_regression(workdir, i)
        try:
            if not ref.survives(workdir):
                caught += 1
        finally:
            restore()

    out["caught_count"] = float(caught)
    return out
