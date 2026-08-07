import ref


def check(workdir):
    from compress.chooser import get_supported_schemes
    out = {"chooser_matches": 0.0}
    match = True
    for arch in ref.ARCHITECTURES:
        gold = set(ref.get_supported_schemes(arch))
        try:
            val = set(get_supported_schemes(arch))
            if gold != val:
                match = False
        except Exception:
            match = False
    out["chooser_matches"] = 1.0 if match else 0.0
    return out
