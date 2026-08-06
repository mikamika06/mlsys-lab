import numpy as np


def simulate_bf16(tensor: np.ndarray) -> np.ndarray:
    f32 = tensor.astype(np.float32)
    u32 = f32.view(np.uint32)
    u32 &= 0xFFFF0000
    return u32.view(np.float32)


def evaluate_cast_error(tensor: np.ndarray, outtype: str) -> float:
    f32 = tensor.astype(np.float32)
    if outtype == "f16":
        cast = f32.astype(np.float16).astype(np.float32)
    elif outtype == "bf16":
        cast = simulate_bf16(f32)
    else:
        raise ValueError(f"Unknown outtype: {outtype}")
    diff = f32 - cast
    return float(np.mean(diff ** 2))
