import ref

def check(workdir):
    from gpucache.capacity import max_model_capacity
    total = 40 * 1024**3
    reserved = 4 * 1024**3
    block_size = 32768
    dtypes = ["float16", "fp8"]
    utils = [0.8, 0.9]
    match = 1
    for dt in dtypes:
        for u in utils:
            want = ref.max_model_capacity(total, reserved, u, block_size, dt)
            try:
                got = max_model_capacity(total, reserved, u, block_size, dt)
            except Exception:
                got = -999
            if got != want:
                match = 0
    return {"capacity_match": float(match)}
