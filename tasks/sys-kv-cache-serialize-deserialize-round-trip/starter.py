import numpy as np


def pack_kv_cache(K: np.ndarray, V: np.ndarray) -> bytes:
    """Serialize a (K, V) KV-cache pair into one self-describing bytes blob.

    See task.md for the exact header layout and byte order.
    """
    raise NotImplementedError('your code here')


def unpack_kv_cache(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `pack_kv_cache`: reconstruct (K, V) exactly from the blob."""
    raise NotImplementedError('your code here')
