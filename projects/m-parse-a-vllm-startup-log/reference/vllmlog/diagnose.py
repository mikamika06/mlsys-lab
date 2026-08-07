def diagnose_garbage(tp, sharding_valid, quant, symptom):
    if symptom != "garbage":
        return "ok"
    if tp == 1:
        return "single_gpu_ok"
    if not sharding_valid:
        return "invalid_sharding"
    return "column_parallel_weight_mismatch"
