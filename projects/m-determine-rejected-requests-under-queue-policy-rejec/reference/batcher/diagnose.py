def attribute_latency(baseline_wait: float, baseline_exec: float, current_wait: float, current_exec: float) -> str:
    wait_diff = current_wait - baseline_wait
    exec_diff = current_exec - baseline_exec
    if wait_diff > exec_diff and wait_diff > 0:
        return "batcher"
    elif exec_diff > wait_diff and exec_diff > 0:
        return "model"
    return "mixed"
