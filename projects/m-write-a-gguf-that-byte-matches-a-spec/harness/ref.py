import struct

def make_spec():
    return {
        "version": 3,
        "alignment": 32,
        "kv": [
            ("general.architecture", "llama"),
            ("general.file_type", 1)
        ]
    }

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

def classify_gguf(data):
    if len(data) < 4 or data[:4] != b"GGUF":
        return "invalid_magic"
    version, = struct.unpack("<I", data[4:8])
    if version not in (2, 3):
        return "invalid_version"
    if len(data) < 24:
        return "truncated_header"
    return "valid"

def get_corrupted_fixtures():
    valid = write_gguf(make_spec())
    f1 = b"BADM" + valid[4:]
    f2 = valid[:4] + struct.pack("<I", 99) + valid[8:]
    f3 = valid[:10]
    f4 = b"\x00" * len(valid)
    f5 = valid[:4] + struct.pack("<I", 1) + valid[8:]
    return [
        (f1, "invalid_magic"),
        (f2, "invalid_version"),
        (f3, "truncated_header"),
        (f4, "invalid_magic"),
        (f5, "invalid_version")
    ]

def struct_diff(a, b):
    diffs = []
    if a.get("version") != b.get("version"):
        diffs.append(("version", a.get("version"), b.get("version")))
    return diffs
