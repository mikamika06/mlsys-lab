def generate_comparison_report(before_metrics, after_metrics):
    ppl_diff = after_metrics.get("perplexity", 0.0) - before_metrics.get("perplexity", 0.0)
    recall_diff = after_metrics.get("recall", 0.0) - before_metrics.get("recall", 0.0)
    return {
        "perplexity_diff": float(ppl_diff),
        "recall_diff": float(recall_diff),
        "improved": bool(recall_diff >= 0 and ppl_diff <= 0)
    }
