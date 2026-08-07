def _encode_varint(n):
    res = bytearray()
    while n > 127:
        res.append((n & 127) | 128)
        n >>= 7
    res.append(n)
    return bytes(res)


def _encode_tag(field_num, wire_type):
    return _encode_varint((field_num << 3) | wire_type)


def _encode_string(field_num, s):
    b = s.encode('utf-8')
    return _encode_tag(field_num, 2) + _encode_varint(len(b)) + b


def _encode_msg(field_num, b):
    return _encode_tag(field_num, 2) + _encode_varint(len(b)) + b


def _build_mock_model(opsets, nodes):
    out = bytearray()

    for domain, version in opsets.items():
        msg = bytearray()
        if domain != "":
            msg += _encode_string(1, domain)
        msg += _encode_tag(2, 0) + _encode_varint(version)
        out += _encode_msg(8, msg)

    graph_msg = bytearray()
    for node in nodes:
        node_msg = bytearray()
        if 'name' in node:
            node_msg += _encode_string(3, node['name'])
        node_msg += _encode_string(4, node['op_type'])
        if 'doc' in node:
            node_msg += _encode_string(6, node['doc'])
        graph_msg += _encode_msg(5, node_msg)

    if graph_msg:
        out += _encode_msg(7, graph_msg)

    return bytes(out)


MODELS = [
    _build_mock_model({"": 14}, [{"op_type": "MatMul", "name": "m1"}, {"op_type": "Relu"}]),
    _build_mock_model({"ai.onnx": 12, "ai.onnx.contrib": 1}, [{"op_type": "Conv", "doc": "hi"}]),
    _build_mock_model({}, []),
]


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


def _parse_model(model_bytes: bytes):
    ops = {}
    savings = 0
    offset, end = 0, len(model_bytes)
    while offset < end:
        tag, offset = _read_varint(model_bytes, offset)
        field_num, wire_type = tag >> 3, tag & 7
        if field_num == 7 and wire_type == 2:
            msg_len, offset = _read_varint(model_bytes, offset)
            graph_end = offset + msg_len
            while offset < graph_end:
                g_tag, offset = _read_varint(model_bytes, offset)
                g_field, g_wire = g_tag >> 3, g_tag & 7
                if g_field == 5 and g_wire == 2:
                    n_len, offset = _read_varint(model_bytes, offset)
                    node_end = offset + n_len
                    while offset < node_end:
                        f_start = offset
                        n_tag, offset = _read_varint(model_bytes, offset)
                        n_field, n_wire = n_tag >> 3, n_tag & 7
                        if n_field in (3, 6):
                            if n_wire == 2:
                                slen, offset = _read_varint(model_bytes, offset)
                                offset += slen
                            else:
                                offset = _skip_field(model_bytes, offset, n_wire)
                            savings += (offset - f_start)
                        elif n_field == 4 and n_wire == 2:
                            slen, offset = _read_varint(model_bytes, offset)
                            op_type = model_bytes[offset:offset+slen].decode('utf-8')
                            offset += slen
                            ops[op_type] = ops.get(op_type, 0) + 1
                        else:
                            offset = _skip_field(model_bytes, offset, n_wire)
                else:
                    offset = _skip_field(model_bytes, offset, g_wire)
        else:
            offset = _skip_field(model_bytes, offset, wire_type)
    return ops, savings


def scan_ops(model_bytes: bytes) -> dict:
    ops, _ = _parse_model(model_bytes)
    return ops


def estimate_ort_savings(model_bytes: bytes) -> int:
    _, savings = _parse_model(model_bytes)
    return savings
