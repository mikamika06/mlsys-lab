import re

def parse_blocks(log_str):
    gpu_match = re.search(r"#\s*GPU\s*blocks[:\s]*(\d+)", log_str, re.IGNORECASE)
    cpu_match = re.search(r"#\s*CPU\s*blocks[:\s]*(\d+)", log_str, re.IGNORECASE)
    gpu_blocks = int(gpu_match.group(1)) if gpu_match else 0
    cpu_blocks = int(cpu_match.group(1)) if cpu_match else 0
    return {"gpu_blocks": gpu_blocks, "cpu_blocks": cpu_blocks}
