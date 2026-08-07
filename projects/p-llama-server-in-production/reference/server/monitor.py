def health_check(metrics):
    return metrics.get("cpu_ok", True) and metrics.get("mem_ok", True) and metrics.get("api_ok", True)
