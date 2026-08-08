import ref


def check(workdir):
    from modetbl.analyzer import build_profiles
    out = {"profiles_matched": 0.0, "records": float(len(ref.RAW_RECORDS))}
    try:
        got = build_profiles(ref.RAW_RECORDS)
        want = ref.build_profiles(ref.RAW_RECORDS)
        if got == want:
            out["profiles_matched"] = float(len(ref.RAW_RECORDS))
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {str(e)}"
    return out
