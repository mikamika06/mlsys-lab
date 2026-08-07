def tensor_bytes(tensor):
    shape = tensor["shape"]
    nelements = 1
    for dim in shape:
        nelements *= dim
    qtype = tensor["qtype"]
    if qtype == "F32":
        return nelements * 4
    elif qtype == "F16":
        return nelements * 2
    elif qtype == "Q4_K":
        return (nelements // 256) * 144
    elif qtype == "Q4_K_S":
        return (nelements // 256) * 136
    elif qtype == "Q4_K_M":
        return (nelements // 256) * 144
    return nelements * 4


def predict_output_size(inventory):
    total = 0
    for t in inventory["tensors"]:
        total += tensor_bytes(t)
    return total
