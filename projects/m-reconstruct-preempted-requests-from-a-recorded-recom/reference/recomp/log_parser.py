import re

def parse_preempted_requests(log_lines):
    preempted = []
    pattern = re.compile(r"RECOMPUTE.*?request_id=([a-zA-Z0-9_-]+).*?num_tokens=(\d+)")
    for line in log_lines:
        match = pattern.search(line)
        if match:
            req_id = match.group(1)
            num_tokens = int(match.group(2))
            preempted.append({"request_id": req_id, "num_tokens": num_tokens})
    return preempted
