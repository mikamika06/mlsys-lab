import ref

def check(workdir):
    from mlx_mem.instrument import instrument_loop
    out = {"instrument_match": 0.0}
    try:
        got = instrument_loop(4, 1024)
        want = ref.ref_instrument(4, 1024)
        if got == want:
            out["instrument_match"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = str(e)
    return out
