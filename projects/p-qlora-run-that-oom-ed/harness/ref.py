def oracle_account_memory(w, a, o):
    total = w + a + o
    return {"total_mb": total, "valid": total > 0}

def oracle_detect_growth(hist):
    if len(hist) < 2:
        return False
    return (hist[-1] - hist[0]) > 0

def oracle_apply_checkpointing(model):
    return True

def oracle_quantize_optimizer(states):
    return {k: {"quantized": True, "bits": 8} for k, v in states.items()}

def oracle_run_step(model, batch, accum):
    return {"completed": True, "effective_batch": batch * accum}
