from realignment.align import align_tokens
from realignment.metrics import compute_metrics

def evaluate_uad(config):
    mapping = align_tokens(config["draft_tokens"], config["target_tokens"])
    metrics = compute_metrics(config["draft_tokens"], config["target_tokens"], mapping, 1.5)
    return {
        "mapping": mapping,
        "metrics": metrics,
        "worth_it": config["family"] == "same" or len(mapping) >= 3,
        "throughput_ratio": metrics["effective_throughput"]
    }
