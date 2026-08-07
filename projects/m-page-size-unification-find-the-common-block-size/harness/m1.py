import ref


def check(workdir):
    try:
        from kvpage.unify import find_common_block_size
    except Exception as e:
        return {"configs_matched": 0.0, "_note": f"Import error: {e}"}

    out = {"configs_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.find_common_block_size(cfg)
        try:
            got = find_common_block_size(cfg)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"Config {i}: got {got}, want {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"Config {i} raised {type(e).__name__}: {e}"
    out["configs_matched"] = float(ok)
    return out
