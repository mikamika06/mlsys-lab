import struct

def write_gguf(spec):
    buf = bytearray()
    buf.extend(struct.pack("<I", spec["magic"]))
    buf.extend(struct.pack("<I", spec["version"]))
    buf.extend(struct.pack("<Q", len(spec["tensors"])))
    buf.extend(struct.pack("<Q", len(spec["kv"])))
    for k, v in spec["kv"].items():
        k_bytes = k.encode("utf-8")
        buf.extend(struct.pack("<Q", len(k_bytes)))
        buf.extend(k_bytes)
        v_type = v["type"]
        buf.extend(struct.pack("<I", v_type))
        val = v["value"]
        if v_type == 8:
            vb = val.encode("utf-8")
            buf.extend(struct.pack("<Q", len(vb)))
            buf.extend(vb)
        elif v_type == 4:
            buf.extend(struct.pack("<I", val))
        elif v_type == 6:
            buf.extend(struct.pack("<f", val))
        else:
            raise ValueError(f"unknown type {v_type}")
    for t in spec["tensors"]:
        name_bytes = t["name"].encode("utf-8")
        buf.extend(struct.pack("<Q", len(name_bytes)))
        buf.extend(name_bytes)
        buf.extend(struct.pack("<I", len(t["dims"])))
        for d in t["dims"]:
            buf.extend(struct.pack("<Q", d))
        buf.extend(struct.pack("<I", t["type"]))
        buf.extend(struct.pack("<Q", t["offset"]))
    return bytes(buf)
