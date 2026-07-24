import struct

import numpy as np

_DTYPE_CODE = {
    np.dtype("float32"): 0,
    np.dtype("float16"): 1,
    np.dtype("int8"): 2,
    np.dtype("float64"): 3,
}
_CODE_DTYPE = {v: k for k, v in _DTYPE_CODE.items()}

_HEADER_FMT = "<BIIII"  # dtype_code, num_layers, num_heads, seq_len, head_dim


def pack_kv_cache(K: np.ndarray, V: np.ndarray) -> bytes:
    """Serialize a (K, V) KV-cache pair into one self-describing bytes blob.

    Layout (all little-endian): a 17-byte header
    ``struct.pack("<BIIII", dtype_code, num_layers, num_heads, seq_len, head_dim)``
    followed by K's raw C-contiguous bytes, then V's raw C-contiguous bytes.
    ``dtype_code``: 0=float32, 1=float16, 2=int8, 3=float64.
    """
    assert K.shape == V.shape and K.dtype == V.dtype
    layers, heads, seq_len, head_dim = K.shape
    code = _DTYPE_CODE[K.dtype]
    header = struct.pack(_HEADER_FMT, code, layers, heads, seq_len, head_dim)
    return header + np.ascontiguousarray(K).tobytes() + np.ascontiguousarray(V).tobytes()


def unpack_kv_cache(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `pack_kv_cache`: reconstruct (K, V) exactly from the blob."""
    hsz = struct.calcsize(_HEADER_FMT)
    code, layers, heads, seq_len, head_dim = struct.unpack(_HEADER_FMT, blob[:hsz])
    dt = _CODE_DTYPE[code]
    n = layers * heads * seq_len * head_dim
    itemsize = dt.itemsize
    body = blob[hsz:]
    k_bytes = body[: n * itemsize]
    v_bytes = body[n * itemsize : 2 * n * itemsize]
    K = np.frombuffer(k_bytes, dtype=dt).reshape(layers, heads, seq_len, head_dim).copy()
    V = np.frombuffer(v_bytes, dtype=dt).reshape(layers, heads, seq_len, head_dim).copy()
    return K, V
