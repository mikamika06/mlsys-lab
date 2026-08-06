import re

def parse_logs(unsloth_log, hf_log):
    u_vram = float(re.search(r"vram:\s*([0-9.]+)", unsloth_log).group(1))
    h_vram = float(re.search(r"vram:\s*([0-9.]+)", hf_log).group(1))
    u_speed = float(re.search(r"speed:\s*([0-9.]+)", unsloth_log).group(1))
    h_speed = float(re.search(r"speed:\s*([0-9.]+)", hf_log).group(1))
    saved = (h_vram - u_vram) / h_vram * 100.0
    return {
        "unsloth_vram": u_vram,
        "hf_vram": h_vram,
        "unsloth_speed": u_speed,
        "hf_speed": h_speed,
        "vram_saved_pct": saved
    }
