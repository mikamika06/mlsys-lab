def tensor_bytes(shape, ftype):
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    if ftype == 0:
        return num_elements * 4
    elif ftype == 2:
        return num_elements * 1
    elif ftype == 7:
        return (num_elements // 32) * 18
    return num_elements * 4


def model_total_bytes(tensors, ftype, leave_output=True):
    total = 0
    for t in tensors:
        if not leave_output and ("output" in t["name"] or "lm_head" in t["name"]):
            continue
        total += tensor_bytes(t["shape"], ftype)
    return total
