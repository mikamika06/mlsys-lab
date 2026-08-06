import random

def generate_cases():
    cases = []
    rng = random.Random(42)
    for i in range(5):
        is_hang = i % 2 == 0
        logs = [
            "[INFO] vLLM engine initialized",
            "[INFO] Model loaded successfully",
            "[INFO] Handling request batch 0"
        ]
        metrics = [
            {"timestamp": 100, "cpu_util": 45.0, "gpu_util": 80.0, "alive": True},
            {"timestamp": 101, "cpu_util": 50.0, "gpu_util": 85.0, "alive": True}
        ]
        if is_hang:
            logs.append("[WARN] Request queue stalled")
            metrics.append({"timestamp": 102, "cpu_util": 5.0, "gpu_util": 0.0, "alive": True})
            expected = "hang"
        else:
            logs.append("[ERROR] Segmentation fault (core dumped)")
            metrics.append({"timestamp": 102, "cpu_util": 0.0, "gpu_util": 0.0, "alive": False})
            expected = "crash"
        cases.append({
            "id": i,
            "logs": logs,
            "metrics": metrics,
            "expected": expected
        })
    return cases

CASES = generate_cases()

def parse_logs(logs):
    events = []
    for line in logs:
        parts = line.split(" ", 1)
        level = parts[0].strip("[]") if len(parts) > 1 else "INFO"
        msg = parts[1] if len(parts) > 1 else line
        events.append({"level": level, "message": msg})
    return events

def aggregate_metrics(metrics):
    if not metrics:
        return {"avg_cpu": 0.0, "avg_gpu": 0.0, "last_alive": False}
    avg_cpu = sum(m["cpu_util"] for m in metrics) / len(metrics)
    avg_gpu = sum(m["gpu_util"] for m in metrics) / len(metrics)
    last_alive = metrics[-1]["alive"]
    return {"avg_cpu": avg_cpu, "avg_gpu": avg_gpu, "last_alive": last_alive}

def classify_failure(parsed_logs, agg_metrics):
    has_segfault = any("Segmentation fault" in l["message"] for l in parsed_logs)
    if not agg_metrics["last_alive"] or has_segfault:
        return "crash"
    return "hang"
