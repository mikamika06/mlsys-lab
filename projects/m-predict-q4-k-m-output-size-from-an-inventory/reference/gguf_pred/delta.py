def explain_delta(tensor):
    shape = tensor["shape"]
    nelements = 1
    for dim in shape:
        nelements *= dim
    blocks = nelements // 256
    return blocks * 8
