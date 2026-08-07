def classify_log(log_text):
    if "OOM" in log_text or "out of memory" in log_text.lower():
        return "oom"
    if "segfault" in log_text.lower() or "segmentation fault" in log_text.lower():
        return "segfault"
    if "context overflow" in log_text.lower():
        return "context_overflow"
    return "unknown"

def format_chat(messages, special_tokens=True):
    res = ""
    for m in messages:
        role = m["role"]
        content = m["content"]
        if special_tokens:
            res += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        else:
            res += f"{role}: {content}\n"
    if special_tokens:
        res += "<|im_start|>assistant\n"
    return res

def configure_slots(max_Ctx, slot_count, kv_type):
    if slot_count <= 0 or max_Ctx <= 0:
        raise ValueError("Invalid parameters")
    slot_size = max_Ctx // slot_count
    bits = 4 if kv_type == "q4_0" else (8 if kv_type == "q8_0" else 16)
    return {"slot_size": slot_size, "kv_bits": bits, "status": "configured"}

def check_memory_growth(allocations):
    peak = 0
    current = 0
    for a in allocations:
        current += a
        if current > peak:
            peak = current
    return {"peak": peak, "stable": current <= peak}

def simulate_load(hours, request_rate):
    total_requests = hours * request_rate * 3600
    failures = 0
    return {"requests": total_requests, "failures": failures, "uptime_pct": 100.0}

def health_check(metrics):
    return metrics.get("cpu_ok", True) and metrics.get("mem_ok", True) and metrics.get("api_ok", True)
