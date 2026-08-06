import ref


def check(workdir):
    from kvcalc.calc import compute_kv_cache_bytes

    configs = ref.get_test_configs()
    matched = 0
    total = len(configs)
    for i, cfg in enumerate(configs):
        num_ctx = 2048 * (i + 1)
        want = ref.compute_kv_bytes(cfg, num_ctx)
        try:
            got = compute_kv_cache_bytes(cfg, num_ctx)
        except Exception as e:
            return {"exact_bytes_match": 0.0, "_note": f"config {i} raised {e}"}

        if abs(got - want) < 1e-5:
            matched += 1
        else:
            return {"exact_bytes_match": 0.0, "_note": f"config {i}: got {got}, want {want}"}

    return {"exact_bytes_match": 1.0 if matched == total else 0.0}
