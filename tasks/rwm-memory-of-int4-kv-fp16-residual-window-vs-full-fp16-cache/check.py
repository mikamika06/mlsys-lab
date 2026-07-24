import random
import math
import numpy as np

def _reference_bytes(T, d, nbits, R, group_size):
    # quantized part (int4)
    int_part = (nbits / 8.0) * (T - R) * d
    # residual window in fp16
    res_part = 2.0 * R * d
    # groups for scales and zero‑points
    remaining = max(0, T - R)
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    groups = math.ceil(remaining / group_size) if remaining > 0 else 0
    scale_zero_part = groups * (2 + 1)  # 2 bytes for fp16 scale, 1 byte for zero‑points
    return int(round(int_part + res_part + scale_zero_part))

def _full_bytes(T, d):
    return 2 * T * d

def grade(sol, fx) -> dict:
    ok = 1.0
    # generate several random test cases
    for _ in range(20):
        try:
            T = random.randint(1, 200)
            d = random.randint(1, 512)
            nbits = 4
            R = random.randint(0, T - 1) if T > 1 else 0
            group_size = random.randint(1, min(64, max(1, T)))
        except Exception:
            ok = 0.0
            break

        try:
            ref_bytes = _reference_bytes(T, d, nbits, R, group_size)
            full_bytes = _full_bytes(T, d)

            # candidate bytes from student's implementation
            cand_bytes = sol.kv_memory_usage(T, d, nbits=nbits,
                                             R=R, group_size=group_size)
        except Exception:
            ok = 0.0
            break

        if not isinstance(cand_bytes, int):
            ok = 0.0
            break

        # compute compression ratios
        ref_ratio = full_bytes / ref_bytes if ref_bytes != 0 else float("inf")
        cand_ratio = full_bytes / cand_bytes if cand_bytes != 0 else float("inf")

        # compare within relative tolerance
        if abs(ref_ratio - cand_ratio) > 1e-9 * max(1.0, ref_ratio):
            ok = 0.0
            break

    return {"size_ratio": ok}
