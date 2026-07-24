import numpy as np


def serialize_kv_roundtrip(kv: np.ndarray) -> np.ndarray:
    """Round trip a KV cache through the offload serializer."""
    payload = np.asarray(kv, dtype=np.float16).tobytes()
    return np.frombuffer(payload, dtype=np.float16).copy().reshape(kv.shape)
