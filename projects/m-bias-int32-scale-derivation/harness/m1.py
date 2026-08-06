import sys
import numpy as np
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    from quantizer.scales import derive_bias_scale, dequantize_weights, quantize_bias
    
    i_scale, i_z, i_q, w_scales, w_q, b_real = ref.generate_fixtures()
    
    out = {"max_abs_err_bias_scale": 1e9, "max_abs_err_dequant_w": 1e9, "max_abs_err_quant_b": 1e9}
    
    try:
        got_b_scale = derive_bias_scale(i_scale, w_scales)
        want_b_scale = ref.derive_bias_scale(i_scale, w_scales)
        out["max_abs_err_bias_scale"] = float(np.max(np.abs(got_b_scale - want_b_scale)))
    except Exception as e:
        out["_note_b_scale"] = str(e)
        
    try:
        got_w = dequantize_weights(w_q, w_scales)
        want_w = ref.dequantize_weights(w_q, w_scales)
        out["max_abs_err_dequant_w"] = float(np.max(np.abs(got_w - want_w)))
    except Exception as e:
        out["_note_w"] = str(e)
        
    try:
        got_b = quantize_bias(b_real, want_b_scale)
        want_b = ref.quantize_bias(b_real, want_b_scale)
        out["max_abs_err_quant_b"] = float(np.max(np.abs(got_b - want_b)))
    except Exception as e:
        pass
        
    sys.path.pop(0)
    return out
