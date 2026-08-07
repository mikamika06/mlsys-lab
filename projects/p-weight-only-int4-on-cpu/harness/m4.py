import ref
import numpy as np

def check(workdir):
    m = {"group_scale_ok": 0.0}
    try:
        from int4.quant import quantize_weights
        w = ref.get_test_weights((128, 128))
        for gs in [32, 64, 128]:
            packed, scale, shape = quantize_weights(w, group_size=gs)
            if scale.shape[0] == 128 * 128 // gs:
                m["group_scale_ok"] = 1.0
    except Exception:
        pass
    return m
