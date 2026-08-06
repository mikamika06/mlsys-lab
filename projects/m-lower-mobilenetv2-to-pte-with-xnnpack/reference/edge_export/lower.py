def lower_model(spec):
    nodes = spec.get("nodes", [])
    out = []
    for n in nodes:
        op = n.get("op")
        target = "xnnpack" if op in ("conv2d", "linear", "relu6", "add") else "cpu_fallback"
        out.append({"name": n["name"], "target": target, "op": op})
    return {"format": "pte", "nodes": out}
