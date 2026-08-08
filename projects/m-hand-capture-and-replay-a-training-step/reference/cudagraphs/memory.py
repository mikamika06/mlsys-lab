def allocate_graph_buffers(operations, inputs, outputs):
    buffers = {}
    addresses = {}
    current_addr = 0x1000000

    for name, shape in inputs.items():
        size = int(sum(shape)) * 8 if shape else 8
        addresses[name] = current_addr
        buffers[name] = {"address": current_addr, "size": size}
        current_addr += size + 256

    for op in operations:
        out_name = op["output"]
        if out_name not in addresses:
            size = 1024
            addresses[out_name] = current_addr
            buffers[out_name] = {"address": current_addr, "size": size}
            current_addr += size + 256

    return buffers


def fix_buffer_overwrites(operations, unsafe_aliases):
    fixed_ops = []
    alias_map = {}

    for op in operations:
        new_op = dict(op)
        new_inputs = [alias_map.get(inp, inp) for inp in op["inputs"]]
        new_op["inputs"] = new_inputs

        out_name = op["output"]
        if out_name in unsafe_aliases:
            safe_out = f"{out_name}_dealiased"
            alias_map[out_name] = safe_out
            new_op["output"] = safe_out
            new_op["requires_copy"] = True

        fixed_ops.append(new_op)

    return fixed_ops
