import numpy as np
from mlsys import scorers

def grade(sol, fx) -> dict:
    try:
        test_arrays = [
            np.array([0.0, -0.0, 1.0, -1.0, 3.1415926, 65504.0], dtype=np.float32),
            np.random.randn(100).astype(np.float32) * 1e5,
            np.array([-np.inf, np.inf, np.nan], dtype=np.float32)
        ]
    except Exception:
        return {"fp16": 0.0, "bf16": 0.0}

    fp16_ok = 1.0
    bf16_ok = 1.0

    for arr in test_arrays:
        try:
            fp16_bits, bf16_bits = sol.encode_fp32_to_fp16_and_bf16(arr)
        except Exception:
            return {"fp16": 0.0, "bf16": 0.0}

        ref_fp16 = arr.astype(np.float16).view(np.uint16)
        ref_bf16 = (arr.view(np.uint32) >> 16).astype(np.uint16)

        if scorers.byte_exact_fraction(fp16_bits, ref_fp16) < 1.0:
            fp16_ok = 0.0
        if scorers.byte_exact_fraction(bf16_bits, ref_bf16) < 1.0:
            bf16_ok = 0.0

    return {"fp16": fp16_ok, "bf16": bf16_ok}
