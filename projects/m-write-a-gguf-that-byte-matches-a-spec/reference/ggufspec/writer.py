"""Writer implementation."""
import struct
import io

def write_gguf_bytes(spec):
    buf = io.BytesIO()
    buf.write(b"GGUF")
    buf.write(struct.pack("<I", spec["version"]))
    buf.write(struct.pack("<Q", len(spec["tensors"])))
    buf.write(struct.pack("<Q", len(spec["kv"])))
    for k, t, v in spec["kv"]:
        k_bytes = k.encode("utf-8")
        buf.write(struct.pack("<Q", len(k_bytes)))
        buf.write(k_bytes)
        buf.write(struct.pack("<I", t))
        if t == 8:
            v_bytes = v.encode("utf-8")
            buf.write(struct.pack("<Q", len(v_bytes)))
            buf.write(v_bytes)
        elif t == 4:
            buf.write(struct.pack("<I", v))
    for name, shape, dtype, data in spec["tensors"]:
        n_bytes = name.encode("utf-8")
        buf.write(struct.pack("<Q", len(n_bytes)))
        buf.write(n_bytes)
        buf.write(struct.pack("<I", len(shape)))
        for dim in shape:
            buf.write(struct.pack("<Q", dim))
        buf.write(struct.pack("<I", dtype))
        buf.write(struct.pack("<Q", 0))
    for _, _, _, data in spec["tensors"]:
        buf.write(data)
    return buf.getvalue()
