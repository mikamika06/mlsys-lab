import math

def compute_shared_e8m0_scale(weights):
    rows = len(weights)
    cols = len(weights[0])
    exponents = [0] * rows
    for i in range(rows):
        max_val = 0.0
        for j in range(cols):
            v = weights[i][j]
            if v < 0.0:
                v = -v
            if j == 0 or v > max_val:
                max_val = v
        val = max_val / 6.0
        if val <= 0.0:
            exponents[i] = 0
        else:
            exponents[i] = int(max(0, math.ceil(math.log2(val))))
    return exponents
