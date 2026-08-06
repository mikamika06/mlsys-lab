from benchparse.parser import parse_report

def compare_hints(latency_text, throughput_text):
    lat_res = parse_report(latency_text)
    tp_res = parse_report(throughput_text)
    return {
        "latency_mode": lat_res,
        "throughput_mode": tp_res,
        "speedup_ratio": tp_res["throughput_fps"] / (lat_res["throughput_fps"] if lat_res["throughput_fps"] else 1.0)
    }
