import struct


def _read_varint(data: bytes, pos: int):
    val = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, pos
        shift += 7
    raise ValueError("Unterminated varint")


def _skip_field(wire_type: int, data: bytes, pos: int):
    if wire_type == 0:
        _, pos = _read_varint(data, pos)
    elif wire_type == 1:
        pos += 8
    elif wire_type == 2:
        length, pos = _read_varint(data, pos)
        pos += length
    elif wire_type == 5:
        pos += 4
    else:
        raise ValueError(f"Unknown wire type {wire_type}")
    return pos


def decode_onnx_graph(data: bytes) -> dict:
    pos = 0
    graph_name = ""
    nodes = []
    initializers = []
    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if field_number == 1 and wire_type == 2:
            length, pos = _read_varint(data, pos)
            sub_data = data[pos:pos+length]
            pos += length
            sub_graph = decode_onnx_graph(sub_data)
            if "name" in sub_graph:
                graph_name = sub_graph["name"]
            if "nodes" in sub_graph:
                nodes.extend(sub_graph["nodes"])
            if "initializers" in sub_graph:
                initializers.extend(sub_graph["initializers"])
        elif field_number == 2 and wire_type == 2:
            length, pos = _read_varint(data, pos)
            graph_data = data[pos:pos+length]
            pos += length
            g_pos = 0
            while g_pos < len(graph_data):
                g_tag, g_pos = _read_varint(graph_data, g_pos)
                g_fn = g_tag >> 3
                g_wt = g_tag & 0x7
                if g_fn == 1 and g_wt == 2:
                    l2, g_pos = _read_varint(graph_data, g_pos)
                    graph_name = graph_data[g_pos:g_pos+l2].decode("utf-8", errors="ignore")
                    g_pos += l2
                elif g_fn == 2 and g_wt == 2:
                    l2, g_pos = _read_varint(graph_data, g_pos)
                    node_data = graph_data[g_pos:g_pos+l2]
                    g_pos += l2
                    node_name = ""
                    op_type = ""
                    inputs = []
                    outputs = []
                    np_pos = 0
                    while np_pos < len(node_data):
                        nt, np_pos = _read_varint(node_data, np_pos)
                        nfn = nt >> 3
                        nwt = nt & 0x7
                        if nfn == 3 and nwt == 2:
                            ll, np_pos = _read_varint(node_data, np_pos)
                            node_name = node_data[np_pos:np_pos+ll].decode("utf-8", errors="ignore")
                            np_pos += ll
                        elif nfn == 4 and nwt == 2:
                            ll, np_pos = _read_varint(node_data, np_pos)
                            op_type = node_data[np_pos:np_pos+ll].decode("utf-8", errors="ignore")
                            np_pos += ll
                        elif nfn == 1 and nwt == 2:
                            ll, np_pos = _read_varint(node_data, np_pos)
                            inputs.append(node_data[np_pos:np_pos+ll].decode("utf-8", errors="ignore"))
                            np_pos += ll
                        elif nfn == 2 and nwt == 2:
                            ll, np_pos = _read_varint(node_data, np_pos)
                            outputs.append(node_data[np_pos:np_pos+ll].decode("utf-8", errors="ignore"))
                            np_pos += ll
                        else:
                            np_pos = _skip_field(nwt, node_data, np_pos)
                    nodes.append({"name": node_name, "op_type": op_type, "inputs": inputs, "outputs": outputs})
                elif g_fn == 5 and g_wt == 2:
                    l2, g_pos = _read_varint(graph_data, g_pos)
                    init_data = graph_data[g_pos:g_pos+l2]
                    g_pos += l2
                    init_name = ""
                    dims = []
                    data_type = 1
                    raw_data = b""
                    ip_pos = 0
                    while ip_pos < len(init_data):
                        it, ip_pos = _read_varint(init_data, ip_pos)
                        ifn = it >> 3
                        iwt = it & 0x7
                        if ifn == 1 and iwt == 2:
                            ll, ip_pos = _read_varint(init_data, ip_pos)
                            init_name = init_data[ip_pos:ip_pos+ll].decode("utf-8", errors="ignore")
                            ip_pos += ll
                        elif ifn == 2 and iwt == 0:
                            dims_val, ip_pos = _read_varint(init_data, ip_pos)
                            dims.append(dims_val)
                        elif ifn == 7 and iwt == 2:
                            ll, ip_pos = _read_varint(init_data, ip_pos)
                            dims_packed = init_data[ip_pos:ip_pos+ll]
                            ip_pos += ll
                            dp = 0
                            while dp < len(dims_packed):
                                dv, dp = _read_varint(dims_packed, dp)
                                dims.append(dv)
                        elif ifn == 3 and iwt == 0:
                            data_type, ip_pos = _read_varint(init_data, ip_pos)
                        elif ifn == 4 and iwt == 2:
                            ll, ip_pos = _read_varint(init_data, ip_pos)
                            raw_data = init_data[ip_pos:ip_pos+ll]
                            ip_pos += ll
                        else:
                            ip_pos = _skip_field(iwt, init_data, ip_pos)
                    initializers.append({
                        "name": init_name,
                        "dims": dims,
                        "data_type": data_type,
                        "raw_data": raw_data
                    })
                else:
                    g_pos = _skip_field(g_wt, graph_data, g_pos)
        else:
            pos = _skip_field(wire_type, data, pos)
    return {"name": graph_name, "nodes": nodes, "initializers": initializers}
