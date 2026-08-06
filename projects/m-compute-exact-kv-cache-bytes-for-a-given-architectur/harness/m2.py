import ref


def check(workdir):
    from kvcalc.gpu import find_max_gpu_context

    configs = ref.get_test_configs()
    matched = 0
    total = len(configs)
    for i, cfg in enumerate(configs):
        weights_bytes = 4 * 1024 * 1024 * 1024
        vram_bytes = 8 * 1024 * 1024 * 1024
        want = ref.find_max_ctx(cfg, weights_bytes, vram_bytes)
        try:
            got = find_max_gpu_context(cfg, weights_bytes, vram_bytes)
        except Exception as e:
            return {"max_ctx_match": 0.0, "_note": f"config {i} raised {e}"}

        if got == want:
            matched += 1
        else:
            return {"max_ctx_match": 0.0, "_note": f"config {i}: got {got}, want {want}"}

    return {"max_ctx_match": 1.0 if matched == total else 0.0}
