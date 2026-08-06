import numpy as np

def compare_quantization(W):
    rows, cols = W.shape

    min_t = float(W[0, 0])
    max_t = float(W[0, 0])
    for r in range(rows):
        for c in range(cols):
            val = float(W[r, c])
            if val < min_t:
                min_t = val
            if val > max_t:
                max_t = val

    range_t = max_t - min_t
    dq_t = np.empty((rows, cols), dtype=np.float64)
    if range_t == 0.0:
        for r in range(rows):
            for c in range(cols):
                dq_t[r, c] = min_t
    else:
        scale_t = range_t / 255.0
        for r in range(rows):
            for c in range(cols):
                q_val = round((float(W[r, c]) - min_t) / scale_t)
                if q_val < 0:
                    q = 0
                elif q_val > 255:
                    q = 255
                else:
                    q = q_val
                dq_t[r, c] = float(q) * scale_t + min_t

    dq_c = np.empty((rows, cols), dtype=np.float64)
    for c in range(cols):
        min_c = float(W[0, c])
        max_c = float(W[0, c])
        for r in range(1, rows):
            val = float(W[r, c])
            if val < min_c:
                min_c = val
            if val > max_c:
                max_c = val
        range_c = max_c - min_c
        if range_c == 0.0:
            for r in range(rows):
                dq_c[r, c] = min_c
        else:
            scale_c = range_c / 255.0
            for r in range(rows):
                q_val = round((float(W[r, c]) - min_c) / scale_c)
                if q_val < 0:
                    q = 0
                elif q_val > 255:
                    q = 255
                else:
                    q = q_val
                dq_c[r, c] = float(q) * scale_c + min_c

    return dq_t, dq_c
