import numpy as np


def serialize_kv_roundtrip(kv: np.ndarray) -> np.ndarray:
    arr = np.asarray(kv)
    payload = arr.tobytes()
    return np.frombuffer(payload, dtype=arr.dtype).copy().reshape(arr.shape)
