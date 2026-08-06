import sys
import numpy as np
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    from quantizer.conv import integer_conv2d
    
    i_scale, i_z, i_q, w_scales, w_q, b_real = ref.generate_fixtures()
    b_scales = ref.derive_bias_scale(i_scale, w_scales)
    b_q = ref.quantize_bias(b_real, b_scales)
    
    out = {"max_abs_err_accum": 1e9}
    try:
        got = integer_conv2d(i_q, i_z, w_q, b_q)
        want = ref.integer_conv2d(i_q, i_z, w_q, b_q)
        out["max_abs_err_accum"] = float(np.max(np.abs(got - want)))
    except Exception as e:
        out["_note"] = str(e)
        
    sys.path.pop(0)
    return out
