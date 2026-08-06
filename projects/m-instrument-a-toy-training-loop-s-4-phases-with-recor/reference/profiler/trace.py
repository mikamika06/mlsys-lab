def detect_unbalanced_ranges(events):
    """Detect unbalanced push/pop ranges in an event log."""
    stack = []
    for idx, evt in enumerate(events):
        op_type = evt.get("type")
        name = evt.get("name")
        if op_type == "push":
            stack.append((name, idx))
        elif op_type == "pop":
            if not stack:
                return {"balanced": False, "error_index": idx}
            top_name, _ = stack.pop()
            if top_name != name:
                return {"balanced": False, "error_index": idx}
        else:
            return {"balanced": False, "error_index": idx}

    if stack:
        _, first_unclosed_idx = stack[0]
        return {"balanced": False, "error_index": first_unclosed_idx}

    return {"balanced": True, "error_index": -1}
