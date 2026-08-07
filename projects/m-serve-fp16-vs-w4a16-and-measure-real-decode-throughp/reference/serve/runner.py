def simulate_decode_step(config: dict) -> dict:
    bs = config["batch_size"]
    q = config["quant_format"]
    base_tok = float(bs * 45.0)
    base_mem = int(bs * 1024 * 1024 * 512)
    if q == "w4a16":
        tok = base_tok * 1.35
        mem = int(base_mem * 0.55)
    else:
        tok = base_tok
        mem = base_mem
    return {"tokens_per_sec": tok, "memory_bytes": mem}
