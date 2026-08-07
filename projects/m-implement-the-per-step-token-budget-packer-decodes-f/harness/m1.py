import ref


def check(workdir):
    from tokenpacker.packer import pack_step

    out = {"packer_matched": 0.0}
    ok = True
    for cfg in ref.CONFIGS:
        want_dec, want_alloc, want_rem = ref.pack_step(cfg["decodes"], cfg["prefills"], cfg["budget"])
        got_dec, got_alloc, got_rem = pack_step(cfg["decodes"], list(cfg["prefills"]), cfg["budget"])
        if got_dec != want_dec or got_alloc != want_alloc or got_rem != want_rem:
            ok = False
            break
    out["packer_matched"] = 1.0 if ok else 0.0
    return out
