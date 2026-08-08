import sys
import numpy as np

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from longctx.analyzer import compare_methods

    m = {"methods_compared": 0.0}
    try:
        res = compare_methods([], [0.1, 0.2], [0.8, 0.9])
        if isinstance(res, dict) and "superior" in res:
            m["methods_compared"] = 2.0
    except Exception:
        pass
    return m
