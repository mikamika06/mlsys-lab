import struct

MAGIC = b"GGUF"

UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)

_FIXED = {
    UINT8: ("<B", 1), INT8: ("<b", 1),
    UINT16: ("<H", 2), INT16: ("<h", 2),
    UINT32: ("<I", 4), INT32: ("<i", 4),
    FLOAT32: ("<f", 4), BOOL: ("<?", 1),
    UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8),
}

TYPE_NAME = {
    UINT8: "UINT8", INT8: "INT8", UINT16: "UINT16", INT16: "INT16",
    UINT32: "UINT32", INT32: "INT32", FLOAT32: "FLOAT32", BOOL: "BOOL",
    STRING: "STRING", ARRAY: "ARRAY", UINT64: "UINT64", INT64: "INT64",
    FLOAT64: "FLOAT64",
}

BLOCK = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 30: (1, 2),
}

GGML_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 30: "BF16",
}


class GGUFError(Exception):
    pass


def _u64(blob, off):
    return struct.unpack_from("<Q", blob, off)[0], off + 8


def _u32(blob, off):
    return struct.unpack_from("<I", blob, off)[0], off + 4


def _string(blob, off):
    n, off = _u64(blob, off)
    if off + n > len(blob):
        raise GGUFError("string runs past end of file")
    return blob[off:off + n].decode("utf-8", "replace"), off + n


def _value(blob, off, kind):
    if kind == STRING:
        return _string(blob, off)
    if kind == ARRAY:
        sub, off = _u32(blob, off)
        n, off = _u64(blob, off)
        out = []
        for _ in range(n):
            v, off = _value(blob, off, sub)
            out.append(v)
        return out, off
    if kind not in _FIXED:
        raise GGUFError("unknown value type %d" % kind)
    fmt, size = _FIXED[kind]
    if off + size > len(blob):
        raise GGUFError("value runs past end of file")
    return struct.unpack_from(fmt, blob, off)[0], off + size


def parse_header(blob):
    if len(blob) < 24:
        raise GGUFError("file shorter than a GGUF header")
    if blob[:4] != MAGIC:
        raise GGUFError("bad magic %r" % blob[:4])
    version, off = _u32(blob, 4)
    tensor_count, off = _u64(blob, off)
    kv_count, off = _u64(blob, off)
    if version != 3:
        raise GGUFError("unsupported GGUF version %d" % version)
    return {"magic": MAGIC.decode(), "version": version,
            "tensor_count": tensor_count, "kv_count": kv_count, "cursor": off}


def parse_kv(blob):
    head = parse_header(blob)
    off = head["cursor"]
    kv, types = {}, {}
    for _ in range(head["kv_count"]):
        key, off = _string(blob, off)
        kind, off = _u32(blob, off)
        if kind == ARRAY:
            sub = struct.unpack_from("<I", blob, off)[0]
            types[key] = "ARRAY[%s]" % TYPE_NAME.get(sub, str(sub))
        else:
            types[key] = TYPE_NAME.get(kind, str(kind))
        val, off = _value(blob, off, kind)
        kv[key] = val
    return {"kv": kv, "types": types, "cursor": off}


def parse_tensor_index(blob):
    meta = parse_kv(blob)
    off = meta["cursor"]
    alignment = meta["kv"].get("general.alignment", 32)
    tensors = []
    for _ in range(parse_header(blob)["tensor_count"]):
        name, off = _string(blob, off)
        ndim, off = _u32(blob, off)
        shape = []
        for _ in range(ndim):
            d, off = _u64(blob, off)
            shape.append(d)
        type_id, off = _u32(blob, off)
        rel, off = _u64(blob, off)
        tensors.append({"name": name, "shape_ggml_order": shape,
                        "ggml_type_id": type_id,
                        "ggml_type": GGML_NAME.get(type_id, str(type_id)),
                        "relative_offset": rel})
    data_start = off
    if data_start % alignment:
        data_start += alignment - (data_start % alignment)
    for t in tensors:
        elems = 1
        for d in t["shape_ggml_order"]:
            elems *= d
        block_elems, block_bytes = BLOCK.get(t["ggml_type_id"], (1, 4))
        if elems % block_elems:
            raise GGUFError("%s: %d elements is not a multiple of the %s block"
                            % (t["name"], elems, t["ggml_type"]))
        t["n_elements"] = elems
        t["n_bytes"] = elems // block_elems * block_bytes
        t["absolute_data_offset"] = data_start + t["relative_offset"]
    return {"alignment": alignment, "data_start": data_start, "tensors": tensors}


def validate(blob):
    problems = []
    try:
        index = parse_tensor_index(blob)
    except GGUFError as e:
        return ["header: %s" % e]
    size = len(blob)
    align = index["alignment"]
    if align & (align - 1):
        problems.append("alignment %d is not a power of two" % align)
    spans = []
    for t in index["tensors"]:
        start = t["absolute_data_offset"]
        end = start + t["n_bytes"]
        if start % align:
            problems.append("%s: data offset %d is not %d-aligned"
                            % (t["name"], start, align))
        if start >= size or end > size:
            problems.append("%s: data [%d,%d) runs past end of file (%d bytes)"
                            % (t["name"], start, end, size))
            continue
        spans.append((start, end, t["name"]))
    spans.sort()
    for i in range(1, len(spans)):
        if spans[i][0] < spans[i - 1][1]:
            problems.append("%s overlaps %s" % (spans[i][2], spans[i - 1][2]))
    return problems


def tensor_bytes(blob, info):
    start = info["absolute_data_offset"]
    return blob[start:start + info["n_bytes"]]
