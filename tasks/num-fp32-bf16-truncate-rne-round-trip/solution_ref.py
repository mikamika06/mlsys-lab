import numpy as np


def fp32_to_bf16_codes(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).view(np.uint32)
    q = x >> np.uint32(16)
    r = x & np.uint32(0xffff)
    round_up = (r > np.uint32(0x8000)) | (
        (r == np.uint32(0x8000)) & ((q & np.uint32(1)) != 0)
    )
    return (q + round_up.astype(np.uint32)).astype(np.uint16)
