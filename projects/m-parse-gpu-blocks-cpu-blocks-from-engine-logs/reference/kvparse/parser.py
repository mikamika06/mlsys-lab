import re


def parse_blocks(log_text: str) -> dict:
    gpu_match = re.search(r"(?:#\s*GPU\s*blocks|gpu_blocks|GPU\s*blocks)\s*[:=]?\s*(\d+)", log_text, re.IGNORECASE)
    cpu_match = re.search(r"(?:#\s*CPU\s*blocks|cpu_blocks|CPU\s*blocks)\s*[:=]?\s*(\d+)", log_text, re.IGNORECASE)
    gpu_blocks = int(gpu_match.group(1)) if gpu_match else 0
    cpu_blocks = int(cpu_match.group(1)) if cpu_match else 0
    return {"gpu_blocks": gpu_blocks, "cpu_blocks": cpu_blocks}
