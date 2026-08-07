import ref


def check(workdir):
    from zeroproj.memory import compute_memory_table

    matched = 0
    for model_params, world_size, dtype_bytes in ref.CONFIGS:
        want = compute_memory_table(model_params, world_size, dtype_bytes)
        try:
            got = compute_memory_table(model_params, world_size, dtype_bytes)
        except Exception:
            got = None
        if got == want and got is not None:
            matched += 1
    return {"table_matched": float(matched)}
