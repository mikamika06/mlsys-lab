import struct

def classify_gguf(data):
    if len(data) < 16:
        return "truncated_header"
    magic = struct.unpack("<I", data[:4])[0]
    if magic != 0x46554747:
        return "invalid_magic"
    try:
        version = struct.unpack("<I", data[4:8])[0]
        if version < 1:
            return "invalid_version"
        n_tensors = struct.unpack("<Q", data[8:16])[0]
        if len(data) > 100000:
            return "oversized"
    except Exception:
        return "malformed_structure"
    return "valid"
