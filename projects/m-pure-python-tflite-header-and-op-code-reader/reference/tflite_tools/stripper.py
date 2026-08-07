import struct


def strip_weights(data: bytes) -> bytes:
    if len(data) < 8:
        return data
    root = struct.unpack_from("<I", data, 0)[0]
    if root + 8 > len(data):
        return data
    new_data = bytearray(data)
    struct.pack_into("<I", new_data, root + 4, 0)
    return bytes(new_data)
