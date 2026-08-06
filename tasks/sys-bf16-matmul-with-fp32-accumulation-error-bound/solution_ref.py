import numpy as np


def _bf16_round(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    bits = (bits + np.uint32(0x8000)) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def bf16_matmul_fp32_accum(A, B):
    a = _bf16_round(A)
    b = _bf16_round(B)
    
    m, k = a.shape
    _, n = b.shape
    
    out = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            s = np.float32(0.0)
            for r in range(k):
                s = np.float32(s + np.float32(a[i, r] * b[r, j]))
            out[i, j] = s
            
    return out
