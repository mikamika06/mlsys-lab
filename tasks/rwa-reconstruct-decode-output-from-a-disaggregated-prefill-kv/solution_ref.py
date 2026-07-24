import io
import numpy as np


def serialize_kv(K: np.ndarray, V: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, K=np.asarray(K, dtype=np.float64), V=np.asarray(V, dtype=np.float64))
    return buffer.getvalue()


def decode_from_kv(Q: np.ndarray, payload: bytes) -> np.ndarray:
    buffer = io.BytesIO(payload)
    data = np.load(buffer)
    K = np.asarray(data["K"], dtype=np.float64)
    V = np.asarray(data["V"], dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    logits = Q @ K.T / np.sqrt(K.shape[1])
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights @ V
