from specfail.classifier import TAXONOMY

def compute_metrics(classifications):
    counts = {t: 0 for t in TAXONOMY}
    for c in classifications:
        cat = c.get("category")
        if cat in counts:
            counts[cat] += 1
    total = len(classifications)
    distribution = {k: (v / total if total > 0 else 0.0) for k, v in counts.items()}
    confidence_score = sum(1.0 for v in counts.values() if v > 0) / len(TAXONOMY)
    return {
        "counts": counts,
        "distribution": distribution,
        "confidence_score": confidence_score
    }
