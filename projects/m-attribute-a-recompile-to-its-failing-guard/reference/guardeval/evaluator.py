def evaluate_guard(guard, meta):
    gtype = guard.get("type")
    if gtype == "shape":
        dim = guard["dim"]
        expected = guard["val"]
        shape = meta.get("shape", [])
        if dim >= len(shape) or shape[dim] != expected:
            return False, f"shape[{dim}] == {expected}"
        return True, None
    elif gtype == "dtype":
        expected = guard["val"]
        actual = meta.get("dtype")
        if actual != expected:
            return False, f"dtype == {expected}"
        return True, None
    elif gtype == "stride":
        dim = guard["dim"]
        expected = guard["val"]
        strides = meta.get("strides", [])
        if dim >= len(strides) or strides[dim] != expected:
            return False, f"strides[{dim}] == {expected}"
        return True, None
    elif gtype == "contiguous":
        expected = guard["val"]
        actual = meta.get("is_contiguous", True)
        if actual != expected:
            return False, f"is_contiguous == {expected}"
        return True, None
    return True, None


def evaluate_graph_guards(guards, meta):
    for guard in guards:
        ok, reason = evaluate_guard(guard, meta)
        if not ok:
            return False, reason
    return True, None
