import struct

import numpy as np

from mlsys.scorers import byte_exact_fraction

_DTYPE_CODE = {
    np.dtype("float32"): 0,
    np.dtype("float16"): 1,
    np.dtype("int8"): 2,
    np.dtype("float64"): 3,
}
_CODE_DTYPE = {v: k for k, v in _DTYPE_CODE.items()}

_HEADER_FMT = "<BIIII"  # dtype_code, num_layers, num_heads, seq_len, head_dim


def _oracle_pack(K, V):
    assert K.shape == V.shape and K.dtype == V.dtype
    layers, heads, seq_len, head_dim = K.shape
    code = _DTYPE_CODE[K.dtype]
    header = struct.pack(_HEADER_FMT, code, layers, heads, seq_len, head_dim)
    return header + np.ascontiguousarray(K).tobytes() + np.ascontiguousarray(V).tobytes()


def _oracle_unpack(blob):
    hsz = struct.calcsize(_HEADER_FMT)
    code, layers, heads, seq_len, head_dim = struct.unpack(_HEADER_FMT, blob[:hsz])
    dt = _CODE_DTYPE[code]
    n = layers * heads * seq_len * head_dim
    itemsize = dt.itemsize
    body = blob[hsz:]
    k_bytes = body[: n * itemsize]
    v_bytes = body[n * itemsize : 2 * n * itemsize]
    K = np.frombuffer(k_bytes, dtype=dt).reshape(layers, heads, seq_len, head_dim)
    V = np.frombuffer(v_bytes, dtype=dt).reshape(layers, heads, seq_len, head_dim)
    return K, V


def _make_case(rng, layers, heads, seq_len, head_dim, dtype):
    dt = np.dtype(dtype)
    shape = (layers, heads, seq_len, head_dim)
    if dt.kind == "f":
        arr = rng.standard_normal(shape).astype(dt)
    else:
        arr = rng.integers(-127, 128, size=shape).astype(dt)
    return arr


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        (2, 4, 8, 16, "float32"),
        (1, 2, 5, 4, "int8"),
        (3, 8, 12, 32, "float16"),
        (1, 1, 1, 1, "float64"),
        (4, 2, 6, 8, "float32"),
    ]

    ok = 1.0
    for layers, heads, seq_len, head_dim, dtype in cases:
        K = _make_case(rng, layers, heads, seq_len, head_dim, dtype)
        V = _make_case(rng, layers, heads, seq_len, head_dim, dtype)
        ref_blob = _oracle_pack(K, V)

        try:
            got_blob = sol.pack_kv_cache(K.copy(), V.copy())
        except Exception:
            return {"byte_exact_fraction": 0.0}

        if not isinstance(got_blob, (bytes, bytearray)):
            return {"byte_exact_fraction": 0.0}
        if len(got_blob) != len(ref_blob):
            ok = 0.0
            continue
        if byte_exact_fraction(ref_blob, bytes(got_blob)) != 1.0:
            ok = 0.0
            continue

        try:
            got_K, got_V = sol.unpack_kv_cache(ref_blob)
        except Exception:
            return {"byte_exact_fraction": 0.0}

        try:
            got_K = np.asarray(got_K)
            got_V = np.asarray(got_V)
        except Exception:
            ok = 0.0
            continue

        if got_K.shape != K.shape or got_V.shape != V.shape:
            ok = 0.0
            continue
        if got_K.dtype != K.dtype or got_V.dtype != V.dtype:
            ok = 0.0
            continue
        if byte_exact_fraction(K.tobytes(), got_K.tobytes()) != 1.0:
            ok = 0.0
            continue
        if byte_exact_fraction(V.tobytes(), got_V.tobytes()) != 1.0:
            ok = 0.0
            continue

    return {"byte_exact_fraction": ok}
