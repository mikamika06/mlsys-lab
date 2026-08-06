def measure_delegated_fraction(model, backend):
    nodes = model.get("nodes", [])
    total = len(nodes)
    if total == 0:
        return 0.0
    delegated = sum(1 for n in nodes if n.get("backend") == backend)
    return float(delegated) / float(total)
