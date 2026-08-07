import ref


def check(workdir):
    from tile_recon.reconstruct import reconstruct_configs
    out = {"configs_matched": 0.0}
    try:
        got = reconstruct_configs(ref.MAX_SMEM, ref.ELEMENT_SIZE, ref.CANDIDATES)
        want = ref.reconstruct_configs(ref.MAX_SMEM, ref.ELEMENT_SIZE, ref.CANDIDATES)
        if got == want:
            out["configs_matched"] = float(len(want))
        else:
            matches = sum(1 for g in got if g in want)
            out["configs_matched"] = float(matches)
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}"
    return out
