def build_kv_dashboard_query(metric_name, window_seconds):
    w = f"{window_seconds}s"
    return f"sum(rate({metric_name}[{w}])) by (model_name, instance)"
