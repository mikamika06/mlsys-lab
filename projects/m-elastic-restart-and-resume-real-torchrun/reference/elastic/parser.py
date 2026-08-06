import re

def parse_nccl_log(log_text: str) -> dict:
    failed_rank = -1
    timeout_op = ""
    match_rank = re.search(r"\[rank\s+(\d+)\]\s+Watchdog caught collective execution timeout", log_text)
    if match_rank:
        failed_rank = int(match_rank.group(1))
    match_op = re.search(r"operation\s+([a-zA-Z0-9_]+)", log_text)
    if match_op:
        timeout_op = match_op.group(1)
    return {"failed_rank": failed_rank, "timeout_op": timeout_op, "has_timeout": failed_rank >= 0}
