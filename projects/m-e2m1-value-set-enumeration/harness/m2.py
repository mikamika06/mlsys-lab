import ref

def check(workdir):
    out = {"matched_quantize": 0.0}
    try:
        from microscale import e2m1
    except ImportError:
        return out

    tensor = [-15.0, -5.0, -2.5, -1.75, -1.5, -1.25, -0.25, 0.0, 0.3, 0.75, 1.25, 1.5, 1.75, 3.5, 10.0]
    configs = [
        (1, False, False),
        (2, True, True),
        (1, True, False)
    ]

    ok = 0
    for cfg in configs:
        want = ref.quantize(tensor, *cfg)
        try:
            got = e2m1.quantize(tensor, *cfg)
        except NotImplementedError:
            return out

        if not isinstance(got, list) or len(want) != len(got):
            out["_note"] = f"quantize result len want {len(want)} got {len(got) if isinstance(got, list) else type(got)}"
            return out

        if want == got:
            ok += 1
        else:
            if "_note" not in out:
                diffs = [(w, g) for w, g in zip(want, got) if w != g]
                out["_note"] = f"quantize {cfg} mismatch: want vs got -> {diffs[:3]}"

    out["matched_quantize"] = float(ok) / float(len(configs))
    return out
