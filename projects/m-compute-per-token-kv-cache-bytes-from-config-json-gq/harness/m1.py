import ref

def check(workdir):
    from kvmem.config import get_bytes_per_token, get_block_size_bytes

    out = {"token_bytes_match": 0.0, "block_bytes_match": 0.0}
    t_ok, b_ok = 0, 0
    total = len(ref.CASES)

    for i, case in enumerate(ref.CASES):
        cfg = case["config"]
        dt = case.get("_override_dtype", case["dtype"])
        bs = case["block_size"]

        want_t = ref.get_bytes_per_token(cfg, dt)
        got_t = get_bytes_per_token(cfg, dt)
        if want_t == got_t:
            t_ok += 1
        elif "_note_t" not in out:
            out["_note_t"] = f"case {i}: token bytes got {got_t}, want {want_t}"

        want_b = ref.get_block_size_bytes(cfg, dt, bs)
        got_b = get_block_size_bytes(cfg, dt, bs)
        if want_b == got_b:
            b_ok += 1
        elif "_note_b" not in out:
            out["_note_b"] = f"case {i}: block bytes got {got_b}, want {want_b}"

    if t_ok == total:
        out["token_bytes_match"] = 1.0
    if b_ok == total:
        out["block_bytes_match"] = 1.0

    return out
