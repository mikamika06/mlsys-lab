import struct
import numpy as np


def parse_gguf_header(data):
    magic, version, tensor_count, kv_count = struct.unpack("<IQQQ", data[:28])
    header = {
        "magic": magic,
        "version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count
    }

    offset = 28
    tensors = []
    for _ in range(tensor_count):
        name_len = struct.unpack("<Q", data[offset:offset+8])[0]
        offset += 8
        name = data[offset:offset+name_len].decode("utf-8")
        offset += name_len
        n_dims = struct.unpack("<I", data[offset:offset+4])[0]
        offset += 4
        dims = []
        for _ in range(n_dims):
            d = struct.unpack("<Q", data[offset:offset+8])[0]
            dims.append(d)
            offset += 8
        qtype = struct.unpack("<I", data[offset:offset+4])[0]
        offset += 4
        t_offset = struct.unpack("<Q", data[offset:offset+8])[0]
        offset += 8

        tensors.append({
            "name": name,
            "n_dims": n_dims,
            "dims": dims,
            "type": qtype,
            "offset": t_offset
        })

    return header, tensors
