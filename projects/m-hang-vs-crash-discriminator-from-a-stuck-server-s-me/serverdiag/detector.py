from serverdiag.parser import parse_logs
from serverdiag.metrics import aggregate_metrics

def classify_failure(logs, metrics):
    parsed = parse_logs(logs)
    agg = aggregate_metrics(metrics)
    has_segfault = any("Segmentation fault" in l["message"] for l in parsed)
    if not agg["last_alive"] or has_segfault:
        return "crash"
    return "hang"
