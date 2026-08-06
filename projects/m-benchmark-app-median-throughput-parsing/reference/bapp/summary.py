import re

def parse_metrics(log_text: str) -> dict:
    fps = 0.0
    median = 0.0
    for line in log_text.splitlines():
        m_fps = re.search(r"Throughput:\s+([\d\.]+)\s+FPS", line)
        if m_fps:
            fps = float(m_fps.group(1))
        m_med = re.search(r"Median:\s+([\d\.]+)\s+ms", line)
        if m_med:
            median = float(m_med.group(1))
    return {"fps": fps, "median": median}

def compare_hints(latency_log: str, throughput_log: str) -> dict:
    lat = parse_metrics(latency_log)
    tput = parse_metrics(throughput_log)
    return {
        "latency_hint_fps": lat["fps"],
        "latency_hint_median": lat["median"],
        "throughput_hint_fps": tput["fps"],
        "throughput_hint_median": tput["median"]
    }
