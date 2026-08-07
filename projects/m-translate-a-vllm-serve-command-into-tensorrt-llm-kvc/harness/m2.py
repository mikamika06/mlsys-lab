import ref


def check(workdir):
    from trtllm_config.memory import compute_cache_bytes

    total = 24 * 1024 * 1024 * 1024
    free = 18 * 1024 * 1024 * 1024
    fraction = 0.8
    want = ref.compute_cache_bytes(total, free, fraction, "free")
    got = compute_cache_bytes(total, free, fraction, "free")
    match = 1.0 if got == want else 0.0
    return {"bytes_match": match}
