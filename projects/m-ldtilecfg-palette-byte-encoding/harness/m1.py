import ref


def check(workdir):
    from amx.config import encode_tilecfg, decode_tilecfg

    out = {"configs_matched": 0.0, "configs": float(len(ref.CONFIG_FIXTURES))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIG_FIXTURES):
        want_bytes = ref.encode_tilecfg(cfg["tiles"], cfg["palette_id"], cfg["start_row"])
        got_bytes = encode_tilecfg(cfg["tiles"], cfg["palette_id"], cfg["start_row"])

        if got_bytes == want_bytes:
            decoded = decode_tilecfg(got_bytes)
            want_decoded = ref.decode_tilecfg(want_bytes)
            if decoded == want_decoded:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"config {i}: decode mismatched"
        elif "_note" not in out:
            out["_note"] = f"config {i}: bytes mismatched. Got {list(got_bytes[:8])}, want {list(want_bytes[:8])}"

    out["configs_matched"] = float(ok)
    return out
