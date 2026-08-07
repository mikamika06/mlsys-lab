def bytes_per_token(params, scheme):
    if scheme in ("W16A16", "FP16"):
        return float(params * 2.0)
    elif scheme == "W8A8":
        return float(params * 1.0)
    elif scheme == "W4A16":
        return float(params * 0.5)
    return float(params * 2.0)
