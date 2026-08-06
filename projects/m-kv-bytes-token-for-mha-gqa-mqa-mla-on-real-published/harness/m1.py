import ref


def check(workdir):
    from kvbytes.calc import calc_bytes_per_token
    out = {"calc_match": 0.0}
    ok = True
    for cfg in ref.CONFIGS:
        if cfg["attn_type"] != "mla":
            want = ref.calc_bytes_per_token(cfg)
            try:
                got = calc_bytes_per_token(cfg)
            except Exception:
                ok = False
                break
            if got != want:
                ok = False
                break
    out["calc_match"] = 1.0 if ok else 0.0
    return out
