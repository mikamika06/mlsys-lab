import numpy as np

def per_tensor_int8_symmetric_fake_quant(x):
    x = np.asarray(x)
    amax = 0.0
    for val in x.flat:
        v = float(val)
        abs_v = v if v >= 0.0 else -v
        if abs_v > amax:
            amax = abs_v

    if amax == 0.0:
        scale = 1.0
    else:
        scale = amax / 127.0

    codes = np.zeros(x.shape, dtype=np.int8)
    dequantized = np.zeros(x.shape, dtype=np.float64)

    for i in range(x.size):
        v = float(x.flat[i])
        q = round(v / scale)
        if q > 127:
            c = 127
        elif q < -127:
            c = -127
        else:
            c = int(q)
        codes.flat[i] = c
        dequantized.flat[i] = float(c) * scale

    return codes, dequantized
