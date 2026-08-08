import ref

def check(workdir):
    from kvmem.solver import predict_num_gpu_blocks, solve_max_model_len

    out = {"blocks_match": 0.0, "len_match": 0.0}
    b_ok, l_ok = 0, 0
    total = len(ref.CASES)

    for i, case in enumerate(ref.CASES):
        cfg = case["config"]
        dt = case.get("_override_dtype", case["dtype"])
        vram = case["total_vram"]
        ws = case["weights_size"]
        util = case["util"]
        bs = case["block_size"]

        want_b = ref.predict_num_gpu_blocks(cfg, dt, vram, ws, util, bs)
        got_b = predict_num_gpu_blocks(cfg, dt, vram, ws, util, bs)
        if want_b == got_b:
            b_ok += 1
        elif "_note_b" not in out:
            out["_note_b"] = f"case {i}: blocks got {got_b}, want {want_b}"

        want_l = ref.solve_max_model_len(cfg, dt, vram, ws, util, bs)
        got_l = solve_max_model_len(cfg, dt, vram, ws, util, bs)
        if want_l == got_l:
            l_ok += 1
        elif "_note_l" not in out:
            out["_note_l"] = f"case {i}: len got {got_l}, want {want_l}"

    if b_ok == total:
        out["blocks_match"] = 1.0
    if l_ok == total:
        out["len_match"] = 1.0

    return out
