import numpy as np


def check(workdir):
    import compress.api as api

    m = {"fp32_ok": 0.0, "pal_ok": 0.0, "uint8_ok": 0.0, "sparse_ok": 0.0, "total_ok": 0.0}
    sd = {
        "a": {"type": "fp32", "data": np.zeros(10)},
        "b": {"type": "palette", "indices": np.zeros(10, dtype=np.uint8), "palette": np.zeros(4, dtype=np.float32)},
        "c": {"type": "uint8", "data": np.zeros(10, dtype=np.uint8), "scale": 1.0, "zp": 0},
        "d": {"type": "sparse", "shape": (10,), "indices": np.zeros(2, dtype=np.int32), "data": np.zeros(2, dtype=np.float32)}
    }

    try:
        sizes, total = api.get_sizes(sd)
        if sizes.get("a") == 40: m["fp32_ok"] = 1.0
        if sizes.get("b") == 26: m["pal_ok"] = 1.0
        if sizes.get("c") == 18: m["uint8_ok"] = 1.0
        if sizes.get("d") == 16: m["sparse_ok"] = 1.0
        if total == 100: m["total_ok"] = 1.0
    except Exception:
        pass

    return m
