import struct

import numpy as np

from .reader import (ARRAY, BOOL, F32, F64, I8, I16, I32, I64, MAGIC, STRING,
                     U8, U16, U32, U64, _FMT)

NP_GGML = {np.dtype(np.float32): 0, np.dtype(np.float16): 1, np.dtype(np.int8): 24,
           np.dtype(np.int16): 25, np.dtype(np.int32): 26, np.dtype(np.int64): 27,
           np.dtype(np.float64): 28}


def _string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _infer(v):
    if isinstance(v, bool):
        return BOOL
    if isinstance(v, str):
        return STRING
    if isinstance(v, int):
        return U32 if 0 <= v < 2 ** 32 else I64
    if isinstance(v, float):
        return F32
    if isinstance(v, (list, tuple)):
        return ARRAY
    raise TypeError(f"cannot store {type(v).__name__} in metadata")


def _value(v, t=None) -> bytes:
    t = _infer(v) if t is None else t
    if t == STRING:
        return _string(v)
    if t == ARRAY:
        if not v:
            return struct.pack("<IQ", U32, 0)
        et = _infer(v[0])
        out = struct.pack("<IQ", et, len(v))
        for item in v:
            out += _value(item, et)
        return out
    return struct.pack(_FMT[t], v)


def _kv(key: str, v) -> bytes:
    t = _infer(v)
    return _string(key) + struct.pack("<I", t) + _value(v, t)


def write(path, arch, metadata, tensors, alignment=32):
    """Header, metadata, tensor table, then the data section — with every tensor
    starting on an `alignment` boundary relative to the start of that section."""
    meta = {"general.architecture": arch, "general.alignment": alignment}
    meta.update(metadata or {})

    arrays = []
    offset = 0
    infos = b""
    for t in tensors:
        a = np.ascontiguousarray(t["data"])
        if a.dtype not in NP_GGML:
            raise TypeError(f"no ggml type for {a.dtype}")
        dims = list(reversed(a.shape))
        infos += _string(t["name"]) + struct.pack("<I", len(dims))
        for d in dims:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<IQ", NP_GGML[a.dtype], offset)
        arrays.append((offset, a))
        offset += a.nbytes
        offset += (-offset) % alignment

    kv = b"".join(_kv(k, v) for k, v in meta.items())
    head = MAGIC + struct.pack("<IQQ", 3, len(tensors), len(meta))
    body = head + kv + infos
    pad = (-len(body)) % alignment

    with open(path, "wb") as f:
        f.write(body)
        f.write(b"\x00" * pad)
        end = 0
        for off, a in arrays:
            f.write(b"\x00" * (off - end))
            f.write(a.tobytes())
            end = off + a.nbytes
    return path
