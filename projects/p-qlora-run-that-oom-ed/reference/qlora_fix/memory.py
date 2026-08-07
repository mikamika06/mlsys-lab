def account_memory(weights_mb, activations_mb, optimizer_mb):
    total = weights_mb + activations_mb + optimizer_mb
    return {"total_mb": total, "valid": total > 0}

def detect_growth(history):
    if len(history) < 2:
        return False
    diffs = [history[i] - history[i-1] for i in range(1, len(history))]
    return sum(diffs) > 0

def apply_checkpointing(model):
    setattr(model, "checkpointing", True)
    return True
