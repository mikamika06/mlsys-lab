import struct


def parse_header(data: bytes) -> dict:
    if len(data) < 64:
        raise ValueError("Data too short for tflite header")
    root_table_offset = struct.unpack_from("<I", data, 0)[0]
    file_identifier = data[4:8].decode("ascii", errors="ignore")
    return {
        "root_table_offset": root_table_offset,
        "file_identifier": file_identifier,
        "size": len(data),
    }


def extract_op_codes(data: bytes) -> list:
    if len(data) < 8:
        return []
    root = struct.unpack_from("<I", data, 0)[0]
    if root + 4 > len(data):
        return []
    code_count = struct.unpack_from("<I", data, root)[0]
    codes = []
    base = root + 4
    for i in range(code_count):
        if base + i * 4 + 4 > len(data):
            break
        val = struct.unpack_from("<I", data, base + i * 4)[0]
        codes.append(val)
    return codes


def attribute_bytes(data: bytes) -> dict:
    header_info = parse_header(data)
    total = len(data)
    metadata_size = min(total, header_info["root_table_offset"] + 128)
    buffers_size = total - metadata_size
    if buffers_size < 0:
        buffers_size = 0
        metadata_size = total
    return {
        "metadata_bytes": metadata_size,
        "buffer_bytes": buffers_size,
        "total_bytes": total,
    }
