import numpy as np
import ml_dtypes


def encode_fp8_e4m3fn(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)


def decode_fp8_e4m3fn(codes: np.ndarray) -> np.ndarray:
    arr = np.asarray(codes, dtype=np.uint8)
    return arr.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
