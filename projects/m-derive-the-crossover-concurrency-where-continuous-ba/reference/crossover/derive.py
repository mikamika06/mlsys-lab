def derive_crossover(case):
    t1 = case["t1"]
    tb = case["tb"]
    tp = case["tp"]
    pl = case["prompt_len"]
    gl = case["gen_len"]

    for c in range(1, 1000):
        time_batch1 = c * (pl * t1 + gl * t1)
        num_batches = (c + 31) // 32
        time_cb = num_batches * tb + (pl + gl) * tp * num_batches
        if time_cb < time_batch1:
            return c
    return 1000
