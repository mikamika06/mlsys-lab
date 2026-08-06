import random

def generate_fixtures():
    random.seed(42)
    raw_pow = "GPU Power: 12500 mW\nCPU Power: 8000 mW\nGPU Frequency: 1333 MHz\n"
    asitop_disp = {"gpu_power_mw": 12500.0, "cpu_power_mw": 8000.0, "gpu_freq_mhz": 1333.0}

    events = [
        {"timestamp": 0.0, "type": "start"},
        {"timestamp": 10.5, "type": "cb_end"},
        {"timestamp": 12.0, "type": "cb_start"},
        {"timestamp": 25.0, "type": "cb_end"},
        {"timestamp": 25.2, "type": "cb_start"},
        {"timestamp": 40.0, "type": "cb_end"}
    ]

    samples = [95.0, 96.0, 94.0, 95.0, 40.0, 38.0, 35.0, 95.0, 96.0]
    return raw_pow, asitop_disp, events, samples

def parse_powermetrics(raw_text):
    res = {}
    for line in raw_text.splitlines():
        if "GPU Power:" in line:
            res["gpu_power_mw"] = float(line.split(":")[1].strip().split()[0])
        elif "CPU Power:" in line:
            res["cpu_power_mw"] = float(line.split(":")[1].strip().split()[0])
        elif "GPU Frequency:" in line:
            res["gpu_freq_mhz"] = float(line.split(":")[1].strip().split()[0])
    return res

def cross_check(raw_text, display_dict):
    parsed = parse_powermetrics(raw_text)
    for k, v in display_dict.items():
        if abs(parsed.get(k, 0.0) - v) > 1e-5:
            return False
    return True

def count_gaps(events, threshold_ms):
    gaps = 0
    sorted_ev = sorted(events, key=lambda x: x["timestamp"])
    for i in range(len(sorted_ev) - 1):
        dur = (sorted_ev[i+1]["timestamp"] - sorted_ev[i]["timestamp"]) * 1000.0
        if dur > threshold_ms:
            gaps += 1
    return gaps

def detect_drop(samples, threshold_ratio=0.5):
    if not samples:
        return -1
    baseline = sum(samples[:3]) / min(3, len(samples))
    for i, s in enumerate(samples):
        if s < baseline * threshold_ratio:
            return i
    return -1
