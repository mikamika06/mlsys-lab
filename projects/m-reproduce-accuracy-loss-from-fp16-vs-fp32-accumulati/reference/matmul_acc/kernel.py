import numpy as np


def compute_matmul_accumulation(a: np.ndarray, b: np.ndarray, use_fp32_acc: bool = True) -> np.ndarray:
    if use_fp32_acc:
        return np.dot(a.astype(np.float32), b.astype(np.float32)).astype(np.float32)
    else:
        return np.dot(a.astype(np.float16), b.astype(np.float16)).astype(np.float16).astype(np.float32)
