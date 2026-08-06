import numpy as np

def analyze_ranges(tensor):
    arr = np.asarray(tensor, dtype=np.float32)
    u = arr.view(np.uint32)
    bf16_bits = u & 0xFFFF0000
    bf16_val = bf16_bits.view(np.float32)
    fp16_overflow = np.abs(arr) > 65504.0
    bf16_overflow = np.abs(arr) > 3.3895314e38
    return {
        "bf16_approx": bf16_val,
        "fp16_overflow": fp16_overflow,
        "bf16_overflow": bf16_overflow
    }
