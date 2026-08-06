import numpy as np


def dynamic_quantize_linear(x: np.ndarray) -> dict:
    """ONNX DynamicQuantizeLinear: derive an asymmetric uint8 scale/zero
    point from x's own (0-including) min/max, round-half-to-even, saturate
    to [0, 255]."""
    x = np.asarray(x, dtype=np.float64)
    qmin, qmax = 0.0, 255.0

    xmin = 0.0
    xmax = 0.0
    
    first = True
    for val in x:
        fval = float(val)
        if first:
            xmin = fval
            xmax = fval
            first = False
        else:
            if fval < xmin:
                xmin = fval
            if fval > xmax:
                xmax = fval

    if 0.0 < xmin:
        xmin = 0.0
    if 0.0 > xmax:
        xmax = 0.0

    y_scale = (xmax - xmin) / (qmax - qmin)
    if y_scale == 0.0:
        y_scale = 1.0

    intermediate_zp = qmin - xmin / y_scale
    
    if intermediate_zp < qmin:
        rounded_zp = qmin
    elif intermediate_zp > qmax:
        rounded_zp = qmax
    else:
        d = round(intermediate_zp)
        if abs(intermediate_zp - d) == 0.5:
            if int(d) % 2 != 0:
                if intermediate_zp > 0:
                    rounded_zp = d - 0.5 if intermediate_zp - d < 0 else d + 0.5
                    # Python round uses round-half-to-even natively
                    rounded_zp = round(intermediate_zp)
                else:
                    rounded_zp = round(intermediate_zp)
            else:
                rounded_zp = round(intermediate_zp)
        else:
            rounded_zp = round(intermediate_zp)
            
    y_zero_point = int(rounded_zp)
    if y_zero_point < int(qmin):
        y_zero_point = int(qmin)
    elif y_zero_point > int(qmax):
        y_zero_point = int(qmax)

    y_list = []
    for val in x:
        fval = float(val)
        scaled = fval / y_scale
        r = round(scaled)
        val_sum = r + y_zero_point
        if val_sum < qmin:
            val_sum = qmin
        elif val_sum > qmax:
            val_sum = qmax
        y_list.append(int(val_sum))

    y = np.array(y_list, dtype=np.uint8)

    return {
        "y": y,
        "y_scale": float(y_scale),
        "y_zero_point": np.uint8(y_zero_point),
    }
