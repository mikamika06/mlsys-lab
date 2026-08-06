import numpy as np


def validate_benchmark_run(run_logs):
    if not run_logs:
        return False, "Empty run log"

    for req in run_logs:
        arr = req.get("arrival_time")
        tok_ts = req.get("token_timestamps", [])

        if arr is None or not isinstance(arr, (int, float)):
            return False, "Invalid arrival time"

        if not tok_ts:
            return False, "Missing token timestamps"

        if tok_ts[0] < arr:
            return False, "TTFT negative"

        if len(tok_ts) > 1:
            diffs = np.diff(tok_ts)
            if np.any(diffs <= 0):
                return False, "Non-positive inter-token timestamp gap"

    return True, "Valid run"
