def rank_sensitivity(tensors):
    scores = []
    for t in tensors:
        score = float(t.get("max_val", 0.0) * abs(t.get("scale", 1.0)))
        scores.append((t["name"], score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores]

def generate_golden(tensors):
    return {t["name"]: float(t.get("max_val", 0.0) * t.get("scale", 1.0)) for t in tensors}
