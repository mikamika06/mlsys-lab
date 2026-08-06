def calc_bytes(ops: list[dict], element_size: int, num_elements: int, fused: bool) -> int:
    if not fused:
        return sum(len(op["inputs"]) + 1 for op in ops) * element_size * num_elements

    created = set(op["output"] for op in ops)
    used = set(inp for op in ops for inp in op["inputs"])

    inputs_count = len(used - created)
    outputs_count = len(created - used)

    return (inputs_count + outputs_count) * element_size * num_elements
