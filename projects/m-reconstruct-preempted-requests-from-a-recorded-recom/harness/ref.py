LOGS = [
    "INFO: vllm scheduler running",
    "WARN: RECOMPUTE warning request_id=req_101 num_tokens=256 batch_idx=3",
    "DEBUG: normal execution continue",
    "WARN: RECOMPUTE warning request_id=req_202 num_tokens=512 batch_idx=4",
    "WARN: RECOMPUTE warning request_id=req_303 num_tokens=128 batch_idx=5"
]

WAITING = [
    {"prompt_tokens": 128},
    {"prompt_tokens": 256},
    {"prompt_tokens": 512}
]

RUNNING = [
    {"seq_len": 1024},
    {"seq_len": 2048}
]

TARGET_RATIO = 0.4

def parse_preempted_requests(log_lines):
    import re
    res = []
    pat = re.compile(r"request_id=([a-zA-Z0-9_-]+).*?num_tokens=(\d+)")
    for l in log_lines:
        m = pat.search(l)
        if m:
            res.append({"request_id": m.group(1), "num_tokens": int(m.group(2))})
    return res

def derive_max_num_batched_tokens(waiting_prompts, running_decodes, target_prefill_ratio):
    total_waiting = sum(p["prompt_tokens"] for p in waiting_prompts)
    total_running = sum(r["seq_len"] for r in running_decodes)
    if not waiting_prompts:
        return total_running
    avg_w = total_waiting / len(waiting_prompts)
    return max(128, int(total_running * (1.0 - target_prefill_ratio) + avg_w * target_prefill_ratio))
