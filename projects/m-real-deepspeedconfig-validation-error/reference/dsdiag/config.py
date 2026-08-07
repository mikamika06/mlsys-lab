def validate_config(cfg):
    tbs = cfg.get("train_batch_size")
    mbs = cfg.get("train_micro_batch_size_per_gpu")
    gas = cfg.get("gradient_accumulation_steps")
    if tbs is None or mbs is None or gas is None:
        return False
    if tbs != mbs * gas * 2:
        return False
    zero = cfg.get("zero_optimization")
    if not zero or "stage" not in zero:
        return False
    return True
