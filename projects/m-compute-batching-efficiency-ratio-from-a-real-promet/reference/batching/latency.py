def decompose_latency(dump_str):
    """Decompose latency into queueing and execution fractions."""
    queues, execs = [], []
    for line in dump_str.splitlines():
        if "nv_inference_queue_duration_us" in line:
            parts = line.strip().split()
            if len(parts) == 2:
                queues.append(float(parts[1]))
        elif "nv_inference_compute_infer_duration_us" in line:
            parts = line.strip().split()
            if len(parts) == 2:
                execs.append(float(parts[1]))
    q_sum = sum(queues) if queues else 0.0
    e_sum = sum(execs) if execs else 0.0
    total = q_sum + e_sum
    if total == 0.0:
        return {"queue_fraction": 0.0, "exec_fraction": 0.0}
    return {"queue_fraction": q_sum / total, "exec_fraction": e_sum / total}
