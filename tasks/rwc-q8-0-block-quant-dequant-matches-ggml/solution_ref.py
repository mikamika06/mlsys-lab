import numpy as np

def q8_0_quantize(x):
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    block_size = 32
    num_blocks = n // block_size
    codes = np.empty_like(x, dtype=np.int8)
    scales = np.empty(num_blocks, dtype=np.float16)
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        absmax = 0.0
        for i in range(start, end):
            val = x[i]
            if val < 0.0:
                val = -val
            if val > absmax:
                absmax = val
        d = absmax / 127.0 if absmax != 0 else 0.0
        scales[b] = np.float16(d)
        for i in range(start, end):
            if d != 0:
                rounded = round(x[i] / d)
                if rounded < -127:
                    c = -127
                elif rounded > 127:
                    c = 127
                else:
                    c = int(rounded)
            else:
                c = 0
            codes[i] = c
    return codes, scales

def q8_0_dequantize(codes, scales):
    block_size = 32
    num_blocks = len(scales)
    x_hat = np.empty_like(codes, dtype=np.float32)
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        d = float(scales[b])
        for i in range(start, end):
            x_hat[i] = float(codes[i]) * d
    return x_hat
