import numpy as np

def _sim(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "fp32":
        return arr.astype(np.float32)
    elif mode == "bf16":
        x = arr.astype(np.float32)
        v = x.view(np.uint32)
        v &= 0xFFFF0000
        return x
    elif mode == "tf32":
        x = arr.astype(np.float32)
        v = x.view(np.uint32)
        v &= 0xFFFFE000
        return x
    elif mode == "fp16":
        return arr.astype(np.float16).astype(np.float32)
    return arr.astype(np.float32)

def compute_matmul_error(a: np.ndarray, b: np.ndarray, precision_mode: str) -> float:
    ref = np.dot(a.astype(np.float64), b.astype(np.float64))
    a_s = _sim(a, precision_mode)
    b_s = _sim(b, precision_mode)
    out = np.dot(a_s.astype(np.float64), b_s.astype(np.float64))
    num = np.linalg.norm(out - ref)
    den = np.linalg.norm(ref) + 1e-12
    return float(num / den)

def simulate_reduction_chain(vectors: list[np.ndarray], precision_mode: str) -> float:
    acc = np.zeros_like(vectors[0], dtype=np.float32)
    for v in vectors:
        vs = _sim(v, precision_mode)
        acc = _sim(acc + vs, precision_mode)
    ref = np.zeros_like(vectors[0], dtype=np.float64)
    for v in vectors:
        ref += v.astype(np.float64)
    num = np.linalg.norm(acc.astype(np.float64) - ref)
    den = np.linalg.norm(ref) + 1e-12
    return float(num / den)
