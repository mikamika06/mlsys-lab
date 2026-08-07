def classify_log(log_text):
    if "OOM" in log_text or "out of memory" in log_text.lower():
        return "oom"
    if "segfault" in log_text.lower() or "segmentation fault" in log_text.lower():
        return "segfault"
    if "context overflow" in log_text.lower():
        return "context_overflow"
    return "unknown"
