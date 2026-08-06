import ref


def check(workdir):
    from compression.ratio import size_ratio

    out = {"ratios_matched": 0.0, "_note": ""}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.compute_size_ratio(
            cfg["weights_count"], cfg["bits"], cfg["group_size"], cfg["scale_bits"]
        )
        try:
            got = size_ratio(
                cfg["weights_count"], cfg["bits"], cfg["group_size"], cfg["scale_bits"]
            )
        except Exception as e:
            out["_note"] = f"config {i} raised {type(e).__name__}: {str(e)[:100]}"
            return out

        if abs(got - want) < 1e-5:
            ok += 1
        else:
            if not out["_note"]:
                out["_note"] = f"config {i}: got {got}, want {want}"
    out["ratios_matched"] = float(ok)
    return out
