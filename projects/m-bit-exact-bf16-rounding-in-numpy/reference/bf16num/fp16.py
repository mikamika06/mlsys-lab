import numpy as np


def fp16_subnormal_mask(x: np.ndarray) -> np.ndarray:
    arr16 = np.asarray(x, dtype=np.float16)
    u16 = arr16.view(np.uint16)
    exp = (u16 >> 10) & 0x1F
    mant = u16 & 0x03FF
    return (exp == 0) & (mant != 0)


def round_fp32_to_fp16(x: np.ndarray, flush_subnormals: bool = False) -> np.ndarray:
    x_f32 = np.asarray(x, dtype=np.float32)
    res = x_f32.astype(np.float16)
    if flush_subnormals:
        sub_mask = fp16_subnormal_mask(res)
        res[sub_mask] = np.copysign(0.0, res[sub_mask])
    return res
