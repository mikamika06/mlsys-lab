import struct

import numpy as np

MAGIC = b"GGUF"

U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)

_FMT = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
        F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d"}

GGML_NP = {0: np.float32, 1: np.float16, 24: np.int8, 25: np.int16,
           26: np.int32, 27: np.int64, 28: np.float64}
GGML_NAME = {0: "F32", 1: "F16", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64"}


class _Cursor:
    def __init__(self, buf):
        self.buf = buf
        self.pos = 0

    def take(self, n):
        out = self.buf[self.pos:self.pos + n]
        if len(out) != n:
            raise ValueError("truncated file")
        self.pos += n
        return out

    def scalar(self, t):
        fmt = _FMT[t]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        n = self.scalar(U64)
        return self.take(n).decode("utf-8")

    def value(self, t):
        if t == STRING:
            return self.string()
        if t == ARRAY:
            et = self.scalar(U32)
            n = self.scalar(U64)
            return [self.value(et) for _ in range(n)]
        if t not in _FMT:
            raise ValueError(f"unknown value type {t}")
        return self.scalar(t)


def _load(path):
    with open(path, "rb") as f:
        return f.read()


def _parse(path):
    buf = _load(path)
    c = _Cursor(buf)
    magic = c.take(4)
    if magic != MAGIC:
        raise ValueError(f"not a GGUF file: magic {magic!r}")
    version = c.scalar(U32)
    tensor_count = c.scalar(U64)
    kv_count = c.scalar(U64)

    meta = {}
    for _ in range(kv_count):
        key = c.string()
        t = c.scalar(U32)
        meta[key] = c.value(t)

    infos = []
    for _ in range(tensor_count):
        name = c.string()
        ndim = c.scalar(U32)
        dims = [c.scalar(U64) for _ in range(ndim)]
        dtype = c.scalar(U32)
        offset = c.scalar(U64)
        infos.append({"name": name, "shape": dims, "dtype": GGML_NAME.get(dtype, str(dtype)),
                      "ggml_type": dtype, "offset": offset})

    alignment = int(meta.get("general.alignment", 32))
    pad = (-c.pos) % alignment
    data_start = c.pos + pad
    return {"buf": buf, "version": version, "tensor_count": tensor_count,
            "kv_count": kv_count, "meta": meta, "infos": infos,
            "alignment": alignment, "data_start": data_start}


def read_header(path):
    p = _parse(path)
    return {"magic": "GGUF", "version": p["version"],
            "tensor_count": p["tensor_count"], "kv_count": p["kv_count"]}


def read_metadata(path):
    return _parse(path)["meta"]


def read_tensor_info(path):
    p = _parse(path)
    return [{k: v for k, v in i.items()} for i in p["infos"]]


def read_tensor_data(path, name):
    p = _parse(path)
    for i in p["infos"]:
        if i["name"] == name:
            n = 1
            for d in i["shape"]:
                n *= d
            dt = np.dtype(GGML_NP[i["ggml_type"]])
            start = p["data_start"] + i["offset"]
            raw = p["buf"][start:start + n * dt.itemsize]
            return np.frombuffer(raw, dtype=dt).reshape(list(reversed(i["shape"])))
    raise KeyError(name)
