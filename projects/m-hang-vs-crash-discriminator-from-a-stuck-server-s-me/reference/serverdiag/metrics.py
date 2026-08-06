def aggregate_metrics(metrics):
    if not metrics:
        return {"avg_cpu": 0.0, "avg_gpu": 0.0, "last_alive": False}
    avg_cpu = sum(m["cpu_util"] for m in metrics) / len(metrics)
    avg_gpu = sum(m["gpu_util"] for m in metrics) / len(metrics)
    last_alive = metrics[-1]["alive"]
    return {"avg_cpu": avg_cpu, "avg_gpu": avg_gpu, "last_alive": last_alive}
