"""Validator implementation."""
import struct

def classify_corruption(data):
    if len(data) < 4:
        return "truncated_file"
    if data[:4] != b"GGUF":
        return "invalid_magic"
    if len(data) < 16:
        return "truncated_file"
    if set(data) == {0}:
        return "all_zeros"
    ver = struct.unpack("<I", data[4:8])[0]
    if ver != 3:
        return "unsupported_version"
    t_count = struct.unpack("<Q", data[8:16])[0]
    if t_count > 1000:
        return "mismatched_tensor_count"
    return "truncated_file"
