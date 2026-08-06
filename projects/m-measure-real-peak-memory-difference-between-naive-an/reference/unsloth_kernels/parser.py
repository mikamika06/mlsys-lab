import re

def parse_training_logs(log_text):
    vram_unsloth = None
    vram_hf = None
    speed_unsloth = None
    speed_hf = None
    for line in log_text.splitlines():
        if "UNSLOTH_VRAM" in line:
            m = re.search(r"([\d\.]+)", line.split("UNSLOTH_VRAM")[1])
            if m:
                vram_unsloth = float(m.group(1))
        elif "HF_VRAM" in line:
            m = re.search(r"([\d\.]+)", line.split("HF_VRAM")[1])
            if m:
                vram_hf = float(m.group(1))
        elif "UNSLOTH_SPEED" in line:
            m = re.search(r"([\d\.]+)", line.split("UNSLOTH_SPEED")[1])
            if m:
                speed_unsloth = float(m.group(1))
        elif "HF_SPEED" in line:
            m = re.search(r"([\d\.]+)", line.split("HF_SPEED")[1])
            if m:
                speed_hf = float(m.group(1))
    return {
        "vram_unsloth": vram_unsloth,
        "vram_hf": vram_hf,
        "speed_unsloth": speed_unsloth,
        "speed_hf": speed_hf
    }
