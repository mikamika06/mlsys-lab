import struct


def generate_mock_tflite(root_offset, op_codes, buffer_size):
    total_size = root_offset + 64 + buffer_size
    buf = bytearray(total_size)
    struct.pack_into("<I", buf, 0, root_offset)
    buf[4:8] = b"TFL3"
    struct.pack_into("<I", buf, root_offset, len(op_codes))
    for i, code in enumerate(op_codes):
        struct.pack_into("<I", buf, root_offset + 4 + i * 4, code)
    struct.pack_into("<I", buf, root_offset + 32, buffer_size)
    return bytes(buf)


MODELS = [
    generate_mock_tflite(16, [1, 3, 5], 1024),
    generate_mock_tflite(32, [2, 8], 2048),
    generate_mock_tflite(24, [0, 4, 6, 9], 512),
]
