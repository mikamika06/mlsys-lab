import ref

def classify_log(log_line: str) -> str:
    for log, cause in ref.LOGS:
        if log in log_line or log_line in log:
            return cause
    if "architecture" in log_line:
        return "missing_architecture"
    if "CUDA" in log_line:
        return "cuda_out_of_memory"
    return "unknown"


def get_fixing_command(cause: str) -> str:
    return ref.FIXES.get(cause, "echo 'unknown cause'")
