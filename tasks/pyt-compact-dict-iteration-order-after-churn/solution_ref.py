def predict_dict_order(ops: list) -> list:
    d = {}
    for op, k in ops:
        if op == "set":
            d[k] = 0
        elif op == "del":
            del d[k]
        else:
            raise ValueError(f"bad op {op!r}")
    return list(d.keys())
