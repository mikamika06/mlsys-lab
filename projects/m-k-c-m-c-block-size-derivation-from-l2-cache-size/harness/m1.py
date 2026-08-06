"""Milestone 1 harness check."""

import ref


def check(workdir):
    from blocking.cache import derive_l2_blocking
    from blocking.registers import derive_register_tile

    out = {"register_tiles_matched": 0.0, "cache_blocks_matched": 0.0}

    reg_ok = 0
    for cfg in ref.REGISTER_CONFIGS:
        want = ref.ref_derive_register_tile(**cfg)
        got = derive_register_tile(**cfg)
        if got == want:
            reg_ok += 1
        elif "_note" not in out:
            out["_note"] = f"register tile mismatch: got {got}, want {want} for {cfg}"

    if reg_ok == len(ref.REGISTER_CONFIGS):
        out["register_tiles_matched"] = 1.0

    cache_ok = 0
    for cfg in ref.CACHE_CONFIGS:
        want = ref.ref_derive_l2_blocking(**cfg)
        got = derive_l2_blocking(**cfg)
        if got == want:
            cache_ok += 1
        elif "_note" not in out:
            out["_note"] = f"cache block mismatch: got {got}, want {want} for {cfg}"

    if cache_ok == len(ref.CACHE_CONFIGS):
        out["cache_blocks_matched"] = 1.0

    return out
