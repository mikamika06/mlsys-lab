import ref
import numpy as np

def check(workdir):
    from ort_tune.io_binding import IOBinder
    m = {"io_binding_ok": 0.0}
    binder = IOBinder()
    t = np.zeros((10, 10))
    res = binder.bind("input_0", t, "cpu")
    if isinstance(res, dict) and res.get("zero_copy") is True:
        m["io_binding_ok"] = 1.0
    return m
