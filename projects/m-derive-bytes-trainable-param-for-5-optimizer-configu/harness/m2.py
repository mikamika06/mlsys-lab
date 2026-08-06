import ref


def check(workdir):
    from optmem.memory import total_full_finetune_memory

    out = {"total_memory_match": 0.0}
    params_bytes = 4000000
    cfg = ref.CONFIGS[1]
    want = ref.total_memory(params_bytes, cfg)
    try:
        got = total_full_finetune_memory(params_bytes, cfg)
    except Exception:
        got = -1.0
    if got == want:
        out["total_memory_match"] = 1.0
    return out
