def parse_processor_column(proc_str):
    lines = [l.strip() for l in proc_str.strip().split("\n") if l.strip()]
    gpu_count = sum(1 for l in lines if "GPU" in l.upper())
    cpu_count = sum(1 for l in lines if "CPU" in l.upper())
    total = gpu_count + cpu_count
    if total == 0:
        return {"gpu_ratio": 0.0, "gpu_layers": 0, "cpu_layers": 0, "total": 0}
    return {"gpu_ratio": gpu_count / total, "gpu_layers": gpu_count, "cpu_layers": cpu_count, "total": total}
