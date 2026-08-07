def _encode_varint(value: int) -> bytes:
    parts = []
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            parts.append(b | 0x80)
        else:
            parts.append(b)
            break
    return bytes(parts)


def _encode_field(field_number: int, wire_type: int, payload: bytes) -> bytes:
    tag = (field_number << 3) | wire_type
    return _encode_varint(tag) + _encode_varint(len(payload)) + payload


def _encode_string(field_number: int, s: str) -> bytes:
    return _encode_field(field_number, 2, s.encode("utf-8"))


def _encode_tensor(name: str, dims: list, data_type: int, raw_data: bytes) -> bytes:
    payload = _encode_string(1, name)
    for d in dims:
        payload += _encode_varint((2 << 3) | 0) + _encode_varint(d)
    payload += _encode_varint((3 << 3) | 0) + _encode_varint(data_type)
    payload += _encode_field(4, 2, raw_data)
    return _encode_field(5, 2, payload)


def _encode_node(name: str, op_type: str, inputs: list, outputs: list) -> bytes:
    payload = _encode_string(3, name) + _encode_string(4, op_type)
    for i in inputs:
        payload += _encode_string(1, i)
    for o in outputs:
        payload += _encode_string(2, o)
    return _encode_field(2, 2, payload)


def _encode_graph(name: str, nodes: list, initializers: list) -> bytes:
    payload = _encode_string(1, name)
    for n in nodes:
        payload += _encode_node(n["name"], n["op_type"], n["inputs"], n["outputs"])
    for init in initializers:
        payload += _encode_tensor(init["name"], init["dims"], init["data_type"], init["raw_data"])
    return _encode_field(2, 2, payload)


SAMPLE_GRAPHS = []
for i in range(3):
    name = f"model_{i}"
    nodes = [{"name": f"conv_{i}", "op_type": "Conv", "inputs": ["X", f"W_{i}"], "outputs": ["Y"]}]
    initializers = [{"name": f"W_{i}", "dims": [16, 16], "data_type": 1, "raw_data": b"\x01" * 1024}]
    raw = _encode_graph(name, nodes, initializers)
    SAMPLE_GRAPHS.append((raw, {"name": name, "nodes": nodes, "initializers": initializers}))
