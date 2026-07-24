import numpy as np


def _bf16_round(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    bits = (bits + np.uint32(0x8000)) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def bf16_matmul_fp32_accum(A, B):
    a = _bf16_round(A)
    b = _bf16_round(B)
    return np.matmul(a, b).astype(np.float32)
