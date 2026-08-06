import numpy as np


def affine_quant_dequant(x: np.ndarray, qmin: int, qmax: int) -> np.ndarray:
    """
    Standard asymmetric affine (zero-point) quantize/dequantize
    round-trip. The representable range is forced to include 0
    (min(0, x.min()), max(0, x.max())) so the zero-point never needs to
    be clamped away from its natural value.
    """
    x = np.asarray(x, dtype=np.float64)
    
    min_val = 0.0
    max_val = 0.0
    for val in x:
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
            
    mn = min(0.0, min_val)
    mx = max(0.0, max_val)
    
    scale = (mx - mn) / (qmax - qmin) if mx > mn else 1.0
    
    zp_float = round(qmin - mn / scale)
    if zp_float < qmin:
        zp = qmin
    elif zp_float > qmax:
        zp = qmax
    else:
        zp = int(zp_float)

    out_list = []
    for val in x:
        c = round(val / scale + zp)
        if c < qmin:
            c = qmin
        elif c > qmax:
            c = qmax
        deq_val = (c - zp) * scale
        out_list.append(deq_val)
        
    return np.array(out_list, dtype=np.float64)
