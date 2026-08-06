def classify_traceback(tb_str):
    tb_lower = tb_str.lower()
    if "dynamo" in tb_lower or "control flow" in tb_lower or "graph break" in tb_lower:
        return "graph_break"
    if "unsupported" in tb_lower or "not supported" in tb_lower:
        return "unsupported_type"
    if "shape mismatch" in tb_lower or "dimension" in tb_lower:
        return "shape_mismatch"
    return "unknown"
