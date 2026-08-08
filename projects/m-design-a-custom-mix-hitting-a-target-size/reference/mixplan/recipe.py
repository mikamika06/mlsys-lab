BITS_PER_FTYPE = {
    "F32": 32.0,
    "F16": 16.0,
    "Q8_0": 8.5,
    "Q4_K": 4.5,
}


def tensor_bytes(shape, ftype):
    p = 1
    for s in shape:
        p *= s
    if len(shape) == 1:
        return p * 4
    bits = BITS_PER_FTYPE.get(ftype, 16.0)
    return int((p * bits + 7) // 8)


def recipe_bytes(config, recipe):
    tot = 0
    for t in config["tensors"]:
        ftype = recipe.get(t["name"], "F16")
        tot += tensor_bytes(t["shape"], ftype)
    return tot
