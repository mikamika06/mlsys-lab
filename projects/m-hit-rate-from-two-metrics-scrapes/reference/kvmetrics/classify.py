def classify_and_aggregate(metric_name):
    if "total" in metric_name or "sum" in metric_name or "count" in metric_name:
        return {"type": "counter", "aggregation": "rate"}
    return {"type": "gauge", "aggregation": "avg"}
