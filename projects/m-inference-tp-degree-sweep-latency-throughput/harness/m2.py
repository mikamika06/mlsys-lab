import ref


def check(workdir):
    from inference.tp_sweep import max_valid_tp_degree, verify_server_log_partitions

    out = {"max_tp_match": 0.0, "logs_verified": 0.0}

    kv_cases = [(8, 32), (4, 16), (3, 8)]
    max_ok = 0
    for kv, attn in kv_cases:
        if max_valid_tp_degree(kv, attn) == ref.max_valid_tp_degree(kv, attn):
            max_ok += 1
    if max_ok == len(kv_cases):
        out["max_tp_match"] = 1.0

    log_ok = 0
    for i, logs in enumerate(ref.SAMPLE_LOGS):
        tp = 4 if i < 2 else 2
        if verify_server_log_partitions(logs, tp) == ref.verify_server_log_partitions(logs, tp):
            log_ok += 1
    if log_ok == len(ref.SAMPLE_LOGS):
        out["logs_verified"] = 1.0

    return out
