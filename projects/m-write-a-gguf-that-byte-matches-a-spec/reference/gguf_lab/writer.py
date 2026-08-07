import struct

def write_gguf(spec):
    buf = bytearray()
    buf.extend(b"GGUF")
    version = spec.get("version", 3)
    buf.extend(struct.pack("<I", version))
    kv = spec.get("kv", {})
    buf.extend(struct.pack("<Q", len(kv)))
    for k, (vtype, vval) in sorted(kv.items()):
        k_bytes = k.encode("utf-8")
        buf.extend(struct.pack("<Q", len(k_bytes)))
        buf.extend(k_bytes)
        buf.extend(struct.pack("<I", vtype))
        if vtype == 5:
            v_bytes = str(vval).encode("utf-8")
            buf.extend(struct.pack("<Q", len(v_bytes)))
            buf.extend(v_bytes)
        elif vtype in (2, 3):
            buf.extend(struct.pack("<I", int(vval)))
        elif vtype == 4:
            buf.extend(struct.pack("<f", float(vval)))
        else:
            buf.extend(struct.pack("<Q", int(vval)))
    tensors = spec.get("tensors", [])
    buf.extend(struct.pack("<Q", len(tensors)))
    for t in tensors:
        name = t["name"].encode("utf-8")
        buf.extend(struct.pack("<Q", len(name)))
        buf.extend(name)
        dims = t.get("dims", [])
        buf.extend(struct.pack("<I", len(dims)))
        for d in dims:
            buf.extend(struct.pack("<Q", int(d)))
        buf.extend(struct.pack("<I", t.get("type", 0)))
        buf.extend(struct.pack("<Q", t.get("offset", 0)))
    return bytes(buf)
