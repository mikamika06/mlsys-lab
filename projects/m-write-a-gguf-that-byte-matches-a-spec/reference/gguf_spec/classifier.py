import struct

def classify_gguf(data):
    if len(data) < 4 or data[:4] != b"GGUF":
        return "invalid_magic"
    version, = struct.unpack("<I", data[4:8])
    if version not in (2, 3):
        return "invalid_version"
    if len(data) < 24:
        return "truncated_header"
    return "valid"
