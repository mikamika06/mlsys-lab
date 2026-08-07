import struct

def write_gguf(spec):
    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<I", spec["version"]))
    buf.extend(struct.pack("<Q", 0))
    buf.extend(struct.pack("<Q", len(spec["kv"])))
    for k, v in spec["kv"]:
        k_bytes = k.encode("utf-8")
        buf.extend(struct.pack("<Q", len(k_bytes)))
        buf.extend(k_bytes)
        if isinstance(v, str):
            buf.extend(struct.pack("<I", 8))
            v_bytes = v.encode("utf-8")
            buf.extend(struct.pack("<Q", len(v_bytes)))
            buf.extend(v_bytes)
        elif isinstance(v, int):
            buf.extend(struct.pack("<I", 4))
            buf.extend(struct.pack("<I", v))
    return bytes(buf)
