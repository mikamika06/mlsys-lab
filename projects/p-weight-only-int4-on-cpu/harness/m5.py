import ref
import numpy as np

def check(workdir):
    m = {"overall_limits_ok": 0.0}
    try:
        from int4.quant import quantize_weights
        w = ref.get_test_weights((128, 128))
        packed, scale, shape = quantize_weights(w, group_size=64)

        size_bytes = packed.nbytes + scale.nbytes
        orig_bytes = w.nbytes
        if size_bytes < orig_bytes * 0.4:
            m["overall_limits_ok"] = 1.0
    except Exception:
        pass
    return m
