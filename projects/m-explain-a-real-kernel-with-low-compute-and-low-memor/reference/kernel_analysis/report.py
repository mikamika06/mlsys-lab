from kernel_analysis.bottleneck import classify_bottleneck
from kernel_analysis.metrics import compute_metrics


def generate_report(k):
    m = compute_metrics(k)
    b = classify_bottleneck(k)
    return {
        "id": k["id"],
        "compute_pct": round(m["compute_pct"], 2),
        "memory_pct": round(m["memory_pct"], 2),
        "bottleneck": b,
    }
