def detect_recompile(history, shape):
    if not history:
        return True
    return shape not in history


def enforce_guard(enabled, recompile_detected):
    if enabled and recompile_detected:
        raise RuntimeError("recompile prohibited")
    return True
