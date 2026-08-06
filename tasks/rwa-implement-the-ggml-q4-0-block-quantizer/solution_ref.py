import numpy as np

def q4_0_quantize(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    blocks = x.reshape(-1, 32)
    n_blocks = blocks.shape[0]
    scales = np.empty(n_blocks, dtype=np.float16)
    codes = np.empty((n_blocks, 16), dtype=np.uint8)

    for b in range(n_blocks):
        block = blocks[b]
        
        max_abs = 0.0
        for i in range(32):
            val = float(block[i])
            if val < 0.0:
                val = -val
            if val > max_abs:
                max_abs = val
                
        d = max_abs / -8.0
        scales[b] = np.float16(d)
        
        for i in range(16):
            v0 = float(block[2 * i])
            scaled0 = v0 / d if d != 0.0 else 0.0
            q0 = round(scaled0)
            if q0 < -8:
                q0 = -8
            elif q0 > 7:
                q0 = 7
            nib0 = int(q0) + 8

            v1 = float(block[2 * i + 1])
            scaled1 = v1 / d if d != 0.0 else 0.0
            q1 = round(scaled1)
            if q1 < -8:
                q1 = -8
            elif q1 > 7:
                q1 = 7
            nib1 = int(q1) + 8

            codes[b, i] = nib0 | (nib1 << 4)

    return scales, codes
