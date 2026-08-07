import numpy as np


def check(workdir):
    from flashwrap.wrapper import flash_wrapper

    out = {"wrapper_correct": 0.0, "layout_handled": 0.0}
    try:
        res1 = flash_wrapper(np.zeros((2, 4, 8, 16)), None, None, scale=1.0, qkvpacked=True)
        if res1.get("mode") == "packed":
            out["wrapper_correct"] = 1.0
    except Exception:
        pass

    try:
        res2 = flash_wrapper(np.zeros((2, 4, 8)), np.zeros((2, 4, 8)), np.zeros((2, 4, 8)), scale=1.0, qkvpacked=False)
        if res2.get("mode") == "unpacked":
            out["layout_handled"] = 1.0
    except Exception:
        pass
    return out
