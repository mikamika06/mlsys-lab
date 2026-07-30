SCALE_BYTES = 2
NO_QUANT_KINDS = {"embed", "norm"}


def tensor_bytes(tensor, scheme):
    count = tensor["count"]
    if tensor["kind"] in NO_QUANT_KINDS or scheme["bits"] >= 16:
        return count * 2
    bits = scheme["bits"]
    weight_bytes = -(-(count * bits) // 8)
    group_size = scheme["group_size"]
    groups = 1 if not group_size else -(-count // group_size)
    return weight_bytes + groups * SCALE_BYTES


def disk_size(model, scheme):
    return sum(tensor_bytes(t, scheme) for t in model["tensors"])


def size_table(model, schemes):
    baseline = disk_size(model, schemes[0])
    rows = []
    for scheme in schemes:
        b = disk_size(model, scheme)
        rows.append({
            "scheme": scheme["name"],
            "bits": scheme["bits"],
            "bytes": b,
            "ratio": b / baseline,
        })
    return rows
