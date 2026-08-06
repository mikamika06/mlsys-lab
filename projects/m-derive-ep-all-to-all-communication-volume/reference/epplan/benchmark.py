def measure_ep_vs_tp_throughput(ep_times: list[float], tp_times: list[float], payload_bytes: int) -> dict:
    ep_avg_time = sum(ep_times) / len(ep_times) if ep_times else 1.0
    tp_avg_time = sum(tp_times) / len(tp_times) if tp_times else 1.0
    ep_throughput = payload_bytes / ep_avg_time
    tp_throughput = payload_bytes / tp_avg_time
    ratio = ep_throughput / tp_throughput if tp_throughput > 0 else 0.0
    return {
        "ep_throughput": ep_throughput,
        "tp_throughput": tp_throughput,
        "ratio": ratio
    }
