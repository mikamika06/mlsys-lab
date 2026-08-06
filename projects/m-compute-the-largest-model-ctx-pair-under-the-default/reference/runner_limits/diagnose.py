def diagnose_log(log_text):
    if "Metal buffer allocation failed" in log_text or "IOReturn(-12)" in log_text:
        return "metal_alloc_failure"
    if "Killed: 9" in log_text or "out of memory" in log_text.lower() or "oom" in log_text.lower():
        return "oom_kill"
    if "segmentation fault" in log_text.lower() or "core dumped" in log_text.lower() or "abort" in log_text.lower():
        return "runner_crash"
    return "success"
