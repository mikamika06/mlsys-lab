import os
import numpy as np
import ref

def check(workdir):
    m = {"quant_error_ok": 0.0}
    mod_path = os.path.join(workdir, "ov_engine", "quantizer.py")
    if not os.path.isfile(mod_path):
        return m

    import sys
    sys.path.insert(0, workdir)
    from ov_engine.quantizer import quantize_int8

    calib = ref.get_calibration_set()
    out_file = os.path.join(workdir, "test_int8.xml")
    try:
        res = quantize_int8("dummy", calib, out_file)
        if res and os.path.exists(out_file):
            m["quant_error_ok"] = 1.0
    except Exception:
        pass
    finally:
        if os.path.exists(out_file):
            os.remove(out_file)
    return m
