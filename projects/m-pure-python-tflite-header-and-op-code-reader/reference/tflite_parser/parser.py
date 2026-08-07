import struct


def parse_header(data: bytes):
    if len(data) < 8:
        raise ValueError("Data too short")
    root_table_offset = struct.unpack_from("<I", data, 0)[0]
    file_identifier = data[4:8]
    return {
        "root_table_offset": root_table_offset,
        "file_identifier": file_identifier,
    }


def get_op_codes(data: bytes):
    header = parse_header(data)
    root = header["root_table_offset"]
    if len(data) < root + 4:
        return []
    vtable_offset = root - struct.unpack_from("<i", data, root)[0]
    vtable_size = struct.unpack_from("<H", data, vtable_offset)[0]

    op_codes_field_offset = None
    curr = vtable_offset + 2
    while curr < vtable_offset + vtable_size:
        field_offset = struct.unpack_from("<H", data, curr)[0]
        field_id = struct.unpack_from("<H", data, curr + 2)[0]
        if field_id == 0:
            op_codes_field_offset = field_offset
            break
        curr += 2

    if not op_codes_field_offset:
        return []

    op_codes_table_pos = root + op_codes_field_offset
    op_codes_vector_offset = op_codes_table_pos + struct.unpack_from("<I", data, op_codes_table_pos)[0]
    vector_len = struct.unpack_from("<I", data, op_codes_vector_offset)[0]

    codes = []
    for i in range(vector_len):
        elem_table_pos = op_codes_vector_offset + 4 + (i * 4)
        elem_offset = elem_table_pos + struct.unpack_from("<I", data, elem_table_pos)[0]

        evtable_offset = elem_offset - struct.unpack_from("<i", data, elem_offset)[0]
        evtable_size = struct.unpack_from("<H", data, evtable_offset)[0]

        builtin_code = 0
        ecurr = evtable_offset + 2
        while ecurr < evtable_offset + evtable_size:
            efield_offset = struct.unpack_from("<H", data, ecurr)[0]
            efield_id = struct.unpack_from("<H", data, ecurr + 2)[0]
            if efield_id == 0:
                val_pos = elem_offset + efield_offset
                builtin_code = struct.unpack_from("<i", data, val_pos)[0]
                break
            ecurr += 2
        codes.append({"builtin_code": int(builtin_code)})
    return codes
