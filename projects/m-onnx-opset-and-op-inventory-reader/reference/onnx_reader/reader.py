def _read_varint(data, offset):
    res, shift = 0, 0
    while True:
        b = data[offset]
        offset += 1
        res |= (b & 127) << shift
        if not (b & 128):
            break
        shift += 7
    return res, offset


def _skip_field(data, offset, wire_type):
    if wire_type == 0:
        _, offset = _read_varint(data, offset)
    elif wire_type == 1:
        offset += 8
    elif wire_type == 2:
        length, offset = _read_varint(data, offset)
        offset += length
    elif wire_type == 5:
        offset += 4
    else:
        raise ValueError("Invalid wire type")
    return offset


def scan_opsets(model_bytes: bytes) -> dict:
    opsets = {}
    offset, end = 0, len(model_bytes)
    while offset < end:
        tag, offset = _read_varint(model_bytes, offset)
        field_num, wire_type = tag >> 3, tag & 7
        if field_num == 8 and wire_type == 2:
            msg_len, offset = _read_varint(model_bytes, offset)
            msg_end = offset + msg_len
            domain, version = "", 1
            while offset < msg_end:
                sub_tag, offset = _read_varint(model_bytes, offset)
                sub_field, sub_wire = sub_tag >> 3, sub_tag & 7
                if sub_field == 1 and sub_wire == 2:
                    slen, offset = _read_varint(model_bytes, offset)
                    domain = model_bytes[offset:offset+slen].decode('utf-8')
                    offset += slen
                elif sub_field == 2 and sub_wire == 0:
                    version, offset = _read_varint(model_bytes, offset)
                else:
                    offset = _skip_field(model_bytes, offset, sub_wire)
            opsets[domain] = version
        else:
            offset = _skip_field(model_bytes, offset, wire_type)
    return opsets


def _parse_graph(model_bytes, start, end):
    ops = {}
    savings = 0
    offset = start
    while offset < end:
        tag, offset = _read_varint(model_bytes, offset)
        field_num, wire_type = tag >> 3, tag & 7
        if field_num == 5 and wire_type == 2:
            msg_len, offset = _read_varint(model_bytes, offset)
            node_end = offset + msg_len
            while offset < node_end:
                field_start = offset
                sub_tag, offset = _read_varint(model_bytes, offset)
                sub_field, sub_wire = sub_tag >> 3, sub_tag & 7

                if sub_field in (3, 6):
                    if sub_wire == 2:
                        slen, offset = _read_varint(model_bytes, offset)
                        offset += slen
                    else:
                        offset = _skip_field(model_bytes, offset, sub_wire)
                    savings += (offset - field_start)
                elif sub_field == 4 and sub_wire == 2:
                    slen, offset = _read_varint(model_bytes, offset)
                    op_type = model_bytes[offset:offset+slen].decode('utf-8')
                    offset += slen
                    ops[op_type] = ops.get(op_type, 0) + 1
                else:
                    offset = _skip_field(model_bytes, offset, sub_wire)
        else:
            offset = _skip_field(model_bytes, offset, wire_type)
    return ops, savings


def _parse_model_for_ops_and_savings(model_bytes: bytes):
    offset, end = 0, len(model_bytes)
    while offset < end:
        tag, offset = _read_varint(model_bytes, offset)
        field_num, wire_type = tag >> 3, tag & 7
        if field_num == 7 and wire_type == 2:
            msg_len, offset = _read_varint(model_bytes, offset)
            return _parse_graph(model_bytes, offset, offset + msg_len)
        else:
            offset = _skip_field(model_bytes, offset, wire_type)
    return {}, 0


def scan_ops(model_bytes: bytes) -> dict:
    ops, _ = _parse_model_for_ops_and_savings(model_bytes)
    return ops


def estimate_ort_savings(model_bytes: bytes) -> int:
    _, savings = _parse_model_for_ops_and_savings(model_bytes)
    return savings
