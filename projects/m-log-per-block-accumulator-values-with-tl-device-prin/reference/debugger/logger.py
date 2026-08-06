import re

def parse_device_print(stdout: str) -> dict:
    result = {}
    # Matches lines like: [0, 1, 2] acc_val: 15.5
    pattern = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+)\]\s*([^:]+):\s*([0-9.-]+)")
    for line in stdout.splitlines():
        match = pattern.search(line)
        if match:
            pid = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            val = float(match.group(5))
            result[pid] = val
    return result
