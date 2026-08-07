def count_precisions(data):
    res = {}
    for l in data.get("layers", []):
        p = l.get("precision", "FP32")
        res[p] = res.get(p, 0) + 1
    return res
