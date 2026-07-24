import numpy as np


def pack_bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> uint16 bf16 codes, round-to-nearest-even (branch-free bias trick)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    return ((u + bias) >> np.uint32(16)).astype(np.uint16)


def unpack_bf16(codes: np.ndarray) -> np.ndarray:
    """uint16 bf16 codes -> the float32 values they denote."""
    c = np.ascontiguousarray(codes, dtype=np.uint16).astype(np.uint32)
    return (c << np.uint32(16)).astype(np.uint32).view(np.float32)
