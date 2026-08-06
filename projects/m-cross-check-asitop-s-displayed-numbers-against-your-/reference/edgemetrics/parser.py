def parse_powermetrics(raw_text: str) -> dict:
    res = {}
    for line in raw_text.splitlines():
        if "GPU Power:" in line:
            res["gpu_power_mw"] = float(line.split(":")[1].strip().split()[0])
        elif "CPU Power:" in line:
            res["cpu_power_mw"] = float(line.split(":")[1].strip().split()[0])
        elif "GPU Frequency:" in line:
            res["gpu_freq_mhz"] = float(line.split(":")[1].strip().split()[0])
    return res

def cross_check(raw_text: str, display_dict: dict) -> bool:
    parsed = parse_powermetrics(raw_text)
    for k, v in display_dict.items():
        if abs(parsed.get(k, 0.0) - v) > 1e-5:
            return False
    return True
