import struct
from tflite_parser.parser import parse_header


def attribute_bytes(data: bytes):
    header = parse_header(data)
    total_size = len(data)
    root = header["root_table_offset"]

    if len(data) < root + 4:
        return {"metadata_bytes": total_size, "buffer_bytes": 0, "total": total_size}

    vtable_offset = root - struct.unpack_from("<i", data, root)[0]
    vtable_size = struct.unpack_from("<H", data, vtable_offset)[0]

    buffers_field_offset = None
    curr = vtable_offset + 2
    while curr < vtable_offset + vtable_size:
        field_offset = struct.unpack_from("<H", data, curr)[0]
        field_id = struct.unpack_from("<H", data, curr + 2)[0]
        if field_id == 1:
            buffers_field_offset = field_offset
            break
        curr += 2

    if not buffers_field_offset:
        return {"metadata_bytes": total_size, "buffer_bytes": 0, "total": total_size}

    buffers_table_pos = root + buffers_field_offset
    buffers_vector_offset = buffers_table_pos + struct.unpack_from("<I", data, buffers_table_pos)[0]
    vector_len = struct.unpack_from("<I", data, buffers_vector_offset)[0]

    buffer_bytes = 0
    for i in range(vector_len):
        elem_table_pos = buffers_vector_offset + 4 + (i * 4)
        elem_offset = elem_table_pos + struct.unpack_from("<I", data, elem_table_pos)[0]

        evtable_offset = elem_offset - struct.unpack_from("<i", data, elem_offset)[0]
        evtable_size = struct.unpack_from("<H", data, evtable_offset)[0]

        data_offset = None
        ecurr = evtable_offset + 2
        while ecurr < evtable_offset + evtable_size:
            efield_offset = struct.unpack_from("<H", data, ecurr)[0]
            efield_id = struct.unpack_from("<H", data, ecurr + 2)[0]
            if efield_id == 0:
                data_pos = elem_offset + efield_offset
                data_offset = data_pos + struct.unpack_from("<I", data, data_pos)[0]
                break
            ecurr += 2

        if data_offset is not None and data_offset < len(data):
            data_len = struct.unpack_from("<I", data, data_offset)[0]
            buffer_bytes += 4 + data_len

    metadata_bytes = total_size - buffer_bytes
    return {
        "metadata_bytes": int(metadata_bytes),
        "buffer_bytes": int(buffer_bytes),
        "total": int(total_size)
    }
